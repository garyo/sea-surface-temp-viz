# sea-surface-temp-viz

Python data pipeline that downloads NOAA OISST sea-surface temperature data nightly, computes aggregates, generates equirectangular WebP textures, and exports time-series JSON. Outputs feed the interactive site at https://globe-viz.oberbrunner.com.

## Sister repo

`~/src/globe-viz` — Astro/Solid/Three.js frontend. It consumes our S3 outputs at runtime; we never push to it directly. Both repos share one approved expansion plan at `/Users/garyo/.claude/plans/tranquil-imagining-seal.md` (multi-source: ERA5 + MODIS LST; regional aggregates including Niño 3.4).

## Deploy / runtime

GitHub Actions workflow `.github/workflows/make-images.yml` runs nightly at 13:15 UTC (just after NOAA's 9 AM EDT update). It prunes the last 90 days from cache (catches reanalyses), regenerates graphs/maps/textures, exports time-series JSON, and uploads everything to S3 (`climate-change-assets/sea-surface-temp/`). AWS creds in `.env` (gitignored).

`upload-to-s3.py` regenerates `index.json` from the S3 bucket listing — the bucket is the source of truth for available dates and regions, not the local repo.

## Important behaviors that aren't obvious from the code

- **`sst-data-cache.json` is NOT committed back from CI**. Each run starts from main's cache, prunes 90 days, and refetches. The cache in main can lag months behind reality; refresh locally before testing aggregation changes:

  ```sh
  uv run prune-cache.py --inplace --days 90
  uv run pipeline.py --source oisst --mode graph --dataset sst --out /tmp/x.png
  ```

- **Cache key** format: `YYYY-MM-DD-{sst|anom}` → float (global cosine-lat-weighted average across ±60° latitude). Phase 2 of the plan migrates this to a nested per-source/region/dataset shape — coordinate via the plan file.

- **`export_timeseries.py`** writes one JSON per region into `maps/timeseries/`. `upload-to-s3.py` walks subdirs recursively, so they land at `sea-surface-temp/timeseries/{region}.json` on S3. Don't flatten that layout — `index.json`'s `timeseries.regions` field is regenerated from the directory structure.

## Conventions

- `uv` for everything Python (per global CLAUDE.md), type hints, ruff.
- Don't add a "refresh cache" commit — the cache file changes whenever you run the pipeline locally; only commit it when the change is intentional (e.g. a backfill).
