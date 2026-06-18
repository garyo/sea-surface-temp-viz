# sea-surface-temp-viz

Python data pipeline that downloads climate data nightly (NOAA OISST sea-surface temperature + ECMWF ERA5 reanalysis SST/2m air temp), computes regional aggregates, generates equirectangular WebP textures, and exports time-series JSON. Outputs feed the interactive site at https://globe-viz.oberbrunner.com.

Despite the legacy repo name, this is now a multi-source pipeline. New sources plug in by subclassing `sources.DataSource` (see `sources/oisst.py` and `sources/era5.py` as the reference implementations).

## Sister repo

`~/src/globe-viz` — Astro/Solid/Three.js frontend. It consumes our S3 outputs at runtime; we never push to it directly. Both repos share one approved expansion plan at `/Users/garyo/.claude/plans/tranquil-imagining-seal.md` (multi-source: ERA5 + MODIS LST; regional aggregates including Niño 3.4).

## Deploy / runtime

GitHub Actions workflow `.github/workflows/make-images.yml` runs nightly at 13:15 UTC (just after NOAA's 9 AM EDT update). It prunes the last 90 days from cache (catches reanalyses), regenerates graphs/maps/textures, exports time-series JSON, and uploads everything to S3 (`climate-change-assets/sea-surface-temp/`). AWS creds in `.env` (gitignored).

`upload-to-s3.py` regenerates `index.json` from the S3 bucket listing — the bucket is the source of truth for available dates and regions, not the local repo.

## Important behaviors that aren't obvious from the code

- **`data-cache.json` is persisted to S3, not committed back from CI.** The committed cache is the authority for *historical* dates (curated by deliberate backfill commits) and is frozen well in the past — it can lag months behind reality. *Recent* dates are never committed; instead each CI run restores `s3://climate-change-assets/sea-surface-temp/data-cache.json`, merges it onto the committed one (`scripts/merge_cache.py` — committed wins on shared keys, S3 fills the recent tail), then pushes the result back at the end. This is what makes a successfully-collected recent day durable: before it, recent ERA5 was rebuilt from scratch each run off the ephemeral `era5-archive/`, so any date an upstream outage missed while it sat in the fetch window was lost forever once it aged out (that's how the May–Jun 2026 Niño-3.4 gap became permanent). The daily backfill now retries still-missing dates over a **90-day** window, so a gap self-heals whenever the source republishes. For testing aggregation changes, refresh locally first (the committed cache lags):

  ```sh
  uv run prune-cache.py --inplace --days 90
  uv run pipeline.py --source oisst --mode graph --dataset sst --out /tmp/x.png
  ```

  Note `prune-cache.py` defaults to `--sources oisst`, so it only prunes OISST entries; ERA5 entries are kept regardless of age (they're refilled from the S3 cache + backfill, not refetched wholesale each run).

- **Cache key** format: `YYYY-MM-DD-{source}-{dataset}-{region}` → float (cosine-lat-weighted average over the named region). Source is `oisst` or `era5`; dataset is one of the source's exposed names (`sst`/`anom` for OISST, `sst`/`t2m` for ERA5); region is one of `regions.REGIONS`. `pipeline.py` and `scripts/aggregate_archive.py` both write this same shape.

- **ERA5 archive** lives under `./era5-archive/YYYY/era5-YYYYMMDD.nc` (gitignored). Files are pre-resampled to the OISST 720×1440 grid + zlib-compressed at fetch time (~3 MB/day). `scripts/backfill_era5.py` does an overnight bulk fetch; the daily cron tops it up via `pipeline.py --source era5 --mode texture`. Requires `~/.cdsapirc` locally and the `CDS_API_KEY` GH secret.

- **`export_timeseries.py`** writes one JSON per region into `maps/timeseries/`. `upload-to-s3.py` walks subdirs recursively, so they land at `sea-surface-temp/timeseries/{region}.json` on S3. Don't flatten that layout — `index.json`'s `timeseries.regions` field is regenerated from the directory structure.

## Conventions

- `uv` for everything Python (per global CLAUDE.md), type hints, ruff.
- Don't add a "refresh cache" commit — the cache file changes whenever you run the pipeline locally; only commit it when the change is intentional (e.g. a backfill).
