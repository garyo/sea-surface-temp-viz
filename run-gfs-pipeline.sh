#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# run-gfs-pipeline.sh — one-shot build + publish of the GFS integration.
#
# Does, in dependency order (parallel where independent, serial where not):
#   Stage 1 (parallel):
#     - ERA5 t2m daily-MAX 1971-2000 climatology      (long: ~400 GB from NCAR)
#     - ERA5 t2m daily-MIN 1971-2000 climatology      (long: ~400 GB from NCAR)
#     - re-render ERA5 t2m / t2m_anom with the new palette (local archive)
#     - GFS backfill pass 1: fetch all daily .nc + render temps + mean anomaly
#                            + aggregate regions               (AWS, ~small)
#   Stage 2 (serial, needs the climatologies):
#     - GFS backfill pass 2: render t2m_max_anom / t2m_min_anom (.nc cached → fast)
#   Stage 3 (serial, publish):
#     - export per-region time series, upload climatologies + cache,
#       push ./maps (textures + series) and regenerate index.json
#
# Run from the repo root with AWS credentials in the environment:
#   ./run-gfs-pipeline.sh                 # full backfill from 2021-01-01
#   START=2024-01-01 ./run-gfs-pipeline.sh
#
# Resource note: the two climatology builds run concurrently — ~6 GB RAM peak
# and two NCAR download streams. They each re-download the same hourly ERA5
# (~800 GB total); a single-pass multi-statistic builder would halve that (see
# GFS-INTEGRATION-PLAN.md). Set SERIAL_CLIM=1 to run them one at a time instead.

set -euo pipefail
cd "$(dirname "$0")"   # repo root

# GFS 0.25° archive starts 2021-03-23 (GFSv16); earlier dates 404 and are skipped.
START="${START:-${1:-2021-03-23}}"           # GFS backfill start
# The ERA5 palette re-render overwrites existing production t2m textures, so it
# defaults to the FULL local archive range (consistent palette across all dates,
# not just recent). Set RERENDER_ERA5=0 to skip it (GFS-only publish); only the
# `t2m` temp palette changed (t2m_anom is untouched), so we re-render t2m alone.
RERENDER_ERA5="${RERENDER_ERA5:-1}"
ERA5_RENDER_START="${ERA5_RENDER_START:-1982-01-01}"
BUCKET="climate-change-assets"
PREFIX="sea-surface-temp"
S3="s3://$BUCKET/$PREFIX"
CLIM_DIR="./era5-archive/climatology"
CLIM_MAX="$CLIM_DIR/era5-t2m-max-climatology-1971-2000.nc"
CLIM_MIN="$CLIM_DIR/era5-t2m-min-climatology-1971-2000.nc"
mkdir -p "$CLIM_DIR" ./maps/timeseries

# Background builds checkpoint, so killing leftovers on early exit is safe.
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

echo "### Restore + merge data cache (so export_timeseries keeps OISST/ERA5 series) ###"
if aws s3 cp "$S3/data-cache.json" /tmp/gfs-s3-cache.json 2>/dev/null; then
  uv run python - <<'PY'
import json, pathlib
loc = pathlib.Path("data-cache.json")
L = json.loads(loc.read_text()) if loc.exists() else {}
S = json.loads(pathlib.Path("/tmp/gfs-s3-cache.json").read_text())
# S3 wins on overlap (freshest OISST/ERA5); local-only keys (our GFS) preserved.
merged = {**L, **S}
loc.write_text(json.dumps(merged, sort_keys=True))
print(f"  merged cache: {len(L)} local + {len(S)} s3 -> {len(merged)} entries")
PY
else
  echo "  (no S3 cache found; using local data-cache.json if present)"
fi

echo
echo "### Stage 1 (parallel): climatologies + ERA5 re-render + GFS temps ###"

run_clim() {  # statistic, outfile
  uv run scripts/build_era5_climatology.py --variable t2m --statistic "$1" --out "$2"
}

if [ "${SERIAL_CLIM:-0}" = "1" ]; then
  run_clim max "$CLIM_MAX"; run_clim min "$CLIM_MIN"
  PID_MAX=""; PID_MIN=""
else
  run_clim max "$CLIM_MAX" & PID_MAX=$!
  run_clim min "$CLIM_MIN" & PID_MIN=$!
fi

if [ "$RERENDER_ERA5" = "1" ]; then
  uv run scripts/generate_era5_textures_batch.py \
    --datasets t2m --start "$ERA5_RENDER_START" \
    --no-skip-existing-on-s3 --out ./maps & PID_ERA5=$!
else
  echo "  (ERA5 palette re-render disabled — RERENDER_ERA5=0)"; PID_ERA5=""
fi

uv run scripts/backfill_gfs.py --start "$START" \
  --out ./maps --cache-file ./data-cache.json & PID_GFS=$!

FAIL=0
[ -n "$PID_ERA5" ] && { wait $PID_ERA5 || { echo "❌ ERA5 re-render failed"; FAIL=1; }; }
wait $PID_GFS  || { echo "❌ GFS backfill (temps) failed"; FAIL=1; }
[ -n "$PID_MAX" ] && { wait $PID_MAX || { echo "❌ max climatology failed"; FAIL=1; }; }
[ -n "$PID_MIN" ] && { wait $PID_MIN || { echo "❌ min climatology failed"; FAIL=1; }; }
[ $FAIL -eq 0 ] || { echo "Stage 1 had failures — stopping before publish."; exit 1; }

echo
echo "### Stage 2 (serial): GFS max/min anomalies (.nc cached → fast) ###"
uv run scripts/backfill_gfs.py --start "$START" \
  --out ./maps --cache-file ./data-cache.json \
  --datasets t2m_max_anom,t2m_min_anom

echo
echo "### Stage 3 (serial): export series + publish ###"
uv run export_timeseries.py --out-dir ./maps/timeseries
aws s3 cp "$CLIM_MAX" "$S3/climatology/"
aws s3 cp "$CLIM_MIN" "$S3/climatology/"
aws s3 cp ./data-cache.json "$S3/data-cache.json" --cache-control "no-store"
uv run ./upload-to-s3.py --maps-dir ./maps

echo
echo "### Done. GFS textures + time series published; index.json regenerated. ###"
