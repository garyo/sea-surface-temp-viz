#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Backfill GFS 2 m air-temperature globe textures + region time series.

Unlike the ERA5 batch (which renders from a pre-present local archive), GFS
data isn't local — each day is assembled from eight 3-hourly 2 m-TMP slices on
``s3://noaa-gfs-bdp-pds`` (see sources/gfs.py). So this runs in two phases:

1. **Fetch (parallel).** Build any missing ``gfs-YYYYMMDD.nc`` (daily
   mean/max/min on the OISST grid) from AWS, concurrently — this is the
   network-bound bulk of the work, and each cached ``.nc`` is a natural
   checkpoint (re-runs skip it).
2. **Render + aggregate (serial).** For each cached day, write the
   ``<date>-gfs-<dataset>-equirect.webp`` + ``-metadata.json`` textures (into
   ``./maps``, matching CI) *and* fold area-weighted region means into
   ``data-cache.json`` (the same cache export_timeseries.py reads), from one
   ``get_data_array`` call. matplotlib isn't thread-safe, so this phase is
   serial.

Anomaly datasets are only rendered/aggregated when their ERA5 climatology is
present locally (max/min await scripts/build_era5_climatology.py --statistic),
so a pre-P2 run cleanly produces just the temps + mean anomaly.

After this, the usual upload-to-s3.py pushes ./maps + regenerates index.json,
and export_timeseries.py emits the per-region JSONs.

Usage:
    uv run scripts/backfill_gfs.py --start 2021-01-01            # full backfill
    uv run scripts/backfill_gfs.py --start 2026-06-01 --workers 12
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import regions  # noqa: E402
from pipeline import cache_key, colormap_for  # noqa: E402
from sources.era5 import Era5Source  # noqa: E402
from sources.gfs import GfsSource, GfsFetchError, _build_daily_nc  # noqa: E402

# The noaa-gfs-bdp-pds 0.25° atmos archive begins at the GFSv16 upgrade
# (2021-03-23) — earlier dates have no gfs.tHHz.pgrb2.0p25 under atmos/ and just
# 404 (the backfill skips them, but starting here avoids ~82 wasted lookups).
DEFAULT_START = datetime.date(2021, 3, 23)
S3_INDEX_URL = "https://climate-change-assets.s3.amazonaws.com/sea-surface-temp/index.json"


def daterange(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    one = datetime.timedelta(days=1)
    out, d = [], start
    while d <= end:
        out.append(d)
        d += one
    return out


def fetch_s3_gfs_dates() -> set[str]:
    """Date strings already carrying a GFS texture on S3 (per index.json)."""
    try:
        with urllib.request.urlopen(S3_INDEX_URL, timeout=30) as resp:
            data = json.load(resp)
        return set(data.get("sources", {}).get("gfs", {}).get("dates", []))
    except Exception as e:  # noqa: BLE001
        print(f"⚠  Failed to fetch S3 index ({e}); not skipping anything")
        return set()


def renderable_datasets(source: GfsSource, requested: list[str] | None) -> list[str]:
    """The datasets to produce: requested (or all), minus anomaly datasets whose
    ERA5 climatology isn't built yet (so a pre-P2 run just does temps + mean
    anomaly instead of crashing on the missing max/min climatologies)."""
    ids = requested or list(source.datasets)
    out: list[str] = []
    for ds_id in ids:
        anom = source._ANOM_INFO.get(ds_id)
        if anom is not None:
            clim_var = anom[1]
            if not Era5Source.climatology_path_for(clim_var).exists():
                print(f"  ⏭  {ds_id}: ERA5 {clim_var} climatology missing — skipping")
                continue
        out.append(ds_id)
    return out


def build_phase(
    dates: list[datetime.date],
    archive_root: Path,
    workers: int,
) -> set[datetime.date]:
    """Parallel-build any missing daily .nc from AWS. Returns dates available.

    Uses *processes*, not threads: each day's build runs cfgrib/eccodes and
    netcdf4/HDF5, whose C libraries share global state that isn't thread-safe —
    concurrent threads corrupt each other's reads/writes (HDF5 "not an HDF5
    file" errors). Separate processes isolate that state, and the work is
    network+CPU bound so the GIL would throttle threads anyway.
    """
    missing = [d for d in dates if not GfsSource.archive_path(archive_root, d).exists()]
    have = set(dates) - set(missing)
    print(f"Fetch phase: {len(have)} cached, {len(missing)} to build from AWS "
          f"({workers} worker processes)")
    if not missing:
        return set(dates)

    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_build_daily_nc, d, GfsSource.archive_path(archive_root, d)): d
            for d in missing
        }
        for fut in as_completed(futs):
            d = futs[fut]
            done += 1
            try:
                fut.result()
                have.add(d)
            except GfsFetchError as e:
                print(f"  ⚠  {d}: {e}")
            except Exception as e:  # noqa: BLE001 — surface unexpected failures
                print(f"  ❌ {d}: {type(e).__name__}: {e}")
            if done % 25 == 0 or done == len(missing):
                el = time.time() - started
                rate = done / max(el, 1)
                eta = (len(missing) - done) / max(rate, 1e-3)
                print(f"  [{done}/{len(missing)}] {rate*60:.0f}/min, ETA {eta/60:.1f}m")
    return have


def render_aggregate_phase(
    source: GfsSource,
    dates: list[datetime.date],
    available: set[datetime.date],
    dataset_ids: list[str],
    archive_root: Path,
    out_dir: Path,
    cache: dict[str, float],
    cache_path: Path,
    save_every: int,
) -> tuple[int, int]:
    """Render textures + fold region means into the cache, serially."""
    cmaps = {ds: colormap_for(source.datasets[ds]) for ds in dataset_ids}
    region_ids = regions.region_ids()
    ok = fail = 0
    started = time.time()

    for i, d in enumerate(dates, 1):
        if d not in available:
            continue
        date_str = d.isoformat()
        nc_path = GfsSource.archive_path(archive_root, d)
        try:
            with source.open_local(nc_path) as raw:
                lat_2d, lon_2d = source.latlon_2d(raw)
                for ds_id in dataset_ids:
                    spec = source.datasets[ds_id]
                    data = source.get_data_array(raw, ds_id)
                    cmap, vmin, vmax = cmaps[ds_id]
                    metadata = {
                        "cmap": spec.cmap_def,
                        "title": spec.title_template.format(date=date_str),
                        "dataset": ds_id,
                        "date": date_str,
                        "year": d.year, "month": d.month, "day": d.day,
                        "variable": spec.variable,
                        "statistic": spec.statistic,
                        "kind": spec.kind,
                    }
                    for fn in source.equirect_filenames(ds_id, date_str):
                        out_path = out_dir / fn
                        plt.imsave(out_path, data, cmap=cmap, vmin=vmin, vmax=vmax,
                                   origin="lower")
                        (out_path.parent / (out_path.stem + "-metadata.json")).write_text(
                            json.dumps(metadata)
                        )
                    for rid in region_ids:
                        val = regions.aggregate(data, lat_2d, lon_2d, rid)
                        cache[cache_key(d.year, d.month, d.day, source.id, ds_id, rid)] = val
            ok += 1
        except Exception as e:  # noqa: BLE001 — surface for triage, keep going
            fail += 1
            print(f"  ❌ {date_str}: {type(e).__name__}: {e}")

        if ok and ok % save_every == 0:
            _save_cache(cache, cache_path)
        if i % 50 == 0 or i == len(dates):
            el = time.time() - started
            rate = i / max(el, 1)
            print(f"[{i}/{len(dates)}] ok={ok} fail={fail} ({rate:.1f}/s)")

    _save_cache(cache, cache_path)
    return ok, fail


def _save_cache(cache: dict[str, float], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, sort_keys=True))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=_date, default=DEFAULT_START)
    p.add_argument("--end", type=_date,
                   default=datetime.date.today() - datetime.timedelta(days=1))
    p.add_argument("--archive-root", type=Path, default=GfsSource.archive_root)
    p.add_argument("--out", type=Path, default=Path("./maps"))
    p.add_argument("--cache-file", type=Path, default=Path("./data-cache.json"))
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel AWS day-builds (network-bound)")
    p.add_argument("--save-every", type=int, default=25,
                   help="Flush data-cache.json every N rendered days")
    p.add_argument("--datasets", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=None, help="Comma-separated dataset ids (default: all available)")
    p.add_argument("--skip-existing-on-s3", action="store_true", default=False,
                   help="Skip dates already carrying a GFS texture on S3")
    args = p.parse_args(argv)

    if args.end < args.start:
        print("❌ --end must be >= --start", file=sys.stderr)
        return 2

    source = GfsSource()
    unknown = [d for d in (args.datasets or []) if d not in source.datasets]
    if unknown:
        p.error(f"unknown dataset(s) {unknown}; gfs has {list(source.datasets)}")
    dataset_ids = renderable_datasets(source, args.datasets)
    if not dataset_ids:
        print("Nothing to render (all requested datasets skipped).")
        return 0

    dates = daterange(args.start, args.end)
    if args.skip_existing_on_s3:
        skip = fetch_s3_gfs_dates()
        before = len(dates)
        dates = [d for d in dates if d.isoformat() not in skip]
        print(f"Skipping {before - len(dates)} dates already on S3")

    args.out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, float] = {}
    if args.cache_file.exists():
        cache = json.loads(args.cache_file.read_text())

    print(f"GFS backfill {args.start} → {args.end} ({len(dates)} days)")
    print(f"Datasets: {dataset_ids}")
    print(f"Output: {args.out.resolve()}  Cache: {args.cache_file.resolve()}")
    print()

    available = build_phase(dates, args.archive_root, args.workers)
    print()
    ok, fail = render_aggregate_phase(
        source, dates, available, dataset_ids, args.archive_root,
        args.out, cache, args.cache_file, args.save_every,
    )
    print()
    print(f"Done. Rendered {ok} days, {fail} failed, "
          f"{len(dates) - len(available)} unavailable from AWS.")
    print("Next: upload-to-s3.py (textures + index.json), export_timeseries.py (series).")
    return 0 if fail == 0 else 1


def _date(s: str) -> datetime.date:
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
