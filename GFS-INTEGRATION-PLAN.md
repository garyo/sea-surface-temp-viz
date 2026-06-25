# GFS 2 m air-temperature source — integration plan

Adds NOAA **GFS** as a near-real-time 2 m air-temperature source alongside ERA5,
producing daily **mean / max / min** textures + region time series. GFS fills
ERA5's 6–7-day reporting gap (operational model, updated 4×/day, ~1–2 h lag).

## Why GFS, and its limits

- **Fresh, not authoritative-long.** GFS is an *operational forecast* model, not
  a reanalysis. It is upgraded periodically (resolution/physics), so its multi-year
  record has discontinuities. → GFS is the "today / recent" layer; **ERA5 stays the
  source for the multi-decade trend graphs.** GFS adds a labeled near-real-time tail.
- **Statistic.** GFS `TMP:2 m above ground` is *instantaneous*. We aggregate eight
  3-hourly slices (f000…f021 of the 00Z cycle, covering the UTC day) per cell into:
  - `t2m_mean` — comparable to the ERA5/OISST daily-mean globe.
  - `t2m_max` / `t2m_min` — diurnal extremes (heat stress, A/C demand) a mean hides.
- **Anomaly baseline.** Use the **ERA5 1971–2000 climatology** (mean/max/min), not a
  GFS self-climatology — GFS only exists from 2015 and isn't stationary, so a GFS
  baseline would be short *and* non-comparable to ERA5 anomalies. The small
  GFS-vs-ERA5 mean-state bias is acceptable for visual comparability (this is the
  analogue of Climate Reanalyzer using a CFSR baseline for GFS maps).

## Data access

- **AWS Open Data, single path:** `s3://noaa-gfs-bdp-pds` (anonymous), 0.25° GRIB2,
  real-time + archive. The `atmos/…pgrb2.0p25` layout begins **2021-03-23** (GFSv16);
  earlier dates 404 (older 0.25° to 2015-01-15 lives in NCAR `d084001`, different path).
  Key: `gfs.YYYYMMDD/00/atmos/gfs.t00z.pgrb2.0p25.fFFF` (+ `.idx` sidecar).
  Subset to just `TMP:2 m above ground` via the `.idx` byte ranges → ~1–2 MB/slice
  instead of ~500 MB. boto3 is already a dep; read GRIB with `cfgrib`.
  (NOMADS is avoided — it only keeps ~10 days and rate-limits hard.)

## Phases

- **P0 — deps. ✅ DONE.** Added `cfgrib`, `eccodes`, `eccodeslib` (the binary; the
  pip `eccodes` package is bindings-only and couldn't find libeccodes without it).
- **P1 — source. ✅ DONE + validated.** `sources/gfs.py`: fetch 8 slices, subset 2 m
  TMP via `.idx` byte ranges, reduce to mean/max/min, resample to the OISST
  720×1440 grid, cache `gfs-YYYYMMDD.nc` (Kelvin). Registered in
  `sources/__init__.py`. `DatasetSpec` gained `variable`/`statistic`/`kind`; the
  three statistics are one `t2m` variable along a statistic axis (so globe-viz can
  render a min/mean/max button group, not three flat datasets) — those axes are
  emitted into each texture's `-metadata.json` by `pipeline.py`.
  - Verified 2026-06-18: one AWS build → three textures. Global daily-mean
    −66→41 °C (mean 8.6); min<mean<max everywhere. Sahara box (Algeria/Mali) daily
    **mean 38 °C**, daily **max 47 °C** — corroborates Climate Reanalyzer's "1-day
    Avg" 40 °C heatwave (it's a real average, not a max).
  - `t2m-mean-anom` works as soon as a texture is built (reuses the existing ERA5
    daily-mean t2m climatology); `t2m-max-anom`/`t2m-min-anom` await P2's
    climatology files (they raise a clear FileNotFoundError until then).
- **P2 — max/min climatologies. 🟡 CODE DONE, run pending.** `build_era5_climatology.py`
  gained `--statistic {mean,max,min}`: it reduces the NCAR-mirror hourly ERA5 to a
  daily max/min (instead of mean) per day, then averages across 1971–2000 →
  mean-of-daily-max / -min. Output var `t2m_max_climatology` / `t2m_min_climatology`
  (matches what `sources/gfs.py` loads). Logic validated on synthetic data.
  - **Run (heavy):** two passes, each ~400 GB of t2m hourly from `s3://nsf-ncar-era5`,
    several hours. Wiring for `t2m_max_anom`/`t2m_min_anom` is already in place
    (`era5.CLIMATOLOGY_FILENAMES` + `gfs._ANOM_INFO`); they light up once the files
    land in `era5-archive/climatology/` (and on S3 for CI):
    ```
    uv run scripts/build_era5_climatology.py --variable t2m --statistic max \
        --out ./era5-archive/climatology/era5-t2m-max-climatology-1971-2000.nc
    uv run scripts/build_era5_climatology.py --variable t2m --statistic min \
        --out ./era5-archive/climatology/era5-t2m-min-climatology-1971-2000.nc
    ```
  - Possible optimization: a single-pass multi-statistic mode (compute mean/max/min
    from one download) would halve the bandwidth — not yet implemented.
- **P3 — backfill. 🟡 CODE DONE, run pending.** `scripts/backfill_gfs.py`: phase 1
  parallel-builds missing `gfs-YYYYMMDD.nc` from AWS (each a checkpoint); phase 2
  serially renders `<date>-gfs-<dataset>-equirect.webp` + metadata into `./maps`
  *and* folds area-weighted region means into `data-cache.json` from one
  `get_data_array` call. Anomaly datasets auto-skip until their ERA5 climatology
  exists (so a pre-P2 run does temps + mean anomaly). Validated on one cached day:
  4 datasets rendered, 40 region aggregates, all matching export_timeseries' key
  regex. **Important fix made here:** GFS dataset ids switched hyphen→underscore
  (`t2m_mean`), because the cache key + `export_timeseries.py` only allow
  `[a-z0-9_]` in the dataset token — hyphens would have dropped every GFS series.
  Texture filenames still hyphenate via `equirect_basename`.
  - **Run sequence (heavy):**
    ```
    uv run scripts/backfill_gfs.py --start 2021-01-01      # ~1600 days from AWS
    ./upload-to-s3.py            # pushes ./maps + regenerates index.json
    uv run export_timeseries.py  # writes per-region JSONs (now incl. gfs)
    # then upload output/timeseries/*.json to S3
    ```
  - `upload-to-s3.py` already indexes the generic `<date>-gfs-<dataset>-equirect.webp`
    pattern (normalizing hyphen→underscore), and `index.json`'s `timeseries.sources`
    picks up `gfs` once it appears in the region JSONs → it shows in the frontend
    source list with no client change.
  - Re-render existing ERA5 t2m with the new red-family palette in the same pass
    (`generate_era5_textures_batch.py --datasets t2m,t2m_anom --no-skip-existing-on-s3`).

- **P4 — daily cron.** Add GFS to the recent-days update so "today" stays fresh.
  Small: the daily driver runs `backfill_gfs.py --start <today-3>` (cheap, re-uses
  cached `.nc`) alongside the existing ERA5/OISST steps. [not yet wired]
- **P4 — daily cron.** Add GFS to the recent-days update so "today" stays fresh. [code]
- **P5 — globe-viz UI.** Expose the GFS source + the mean/max/min variable choices in
  the frontend (cmaps travel in per-texture `-metadata.json`; `index.json` already
  lists datasets per source). [code, other repo]

## Running it all

`./run-gfs-pipeline.sh` orchestrates the whole thing (parallel where independent,
serial where dependent): Stage 1 runs the two climatology builds + ERA5 palette
re-render + GFS temp backfill concurrently; Stage 2 renders GFS max/min anomalies
once the climatologies exist (.nc cached → fast); Stage 3 exports series and
publishes (climatologies, cache, `./maps`, `index.json`). It restore-merges the
S3 data cache first so `export_timeseries.py` keeps the OISST/ERA5 series.
`START=YYYY-MM-DD` and `SERIAL_CLIM=1` env vars tune it.

## Dataset ids (S3 stems → `<date>-gfs-<id>-equirect.webp`)

`t2m_mean`, `t2m_max`, `t2m_min`, `t2m_mean_anom`, `t2m_max_anom`, `t2m_min_anom`.
