# sea-surface-temp-viz

Python data pipeline that downloads climate data nightly (NOAA OISST sea-surface temperature + ECMWF ERA5 reanalysis SST/2m air temp), computes regional aggregates, generates equirectangular WebP textures, and exports time-series JSON. Outputs feed the interactive site at https://globe-viz.oberbrunner.com.

Despite the legacy repo name, this is now a multi-source pipeline. New sources plug in by subclassing `sources.DataSource` (see `sources/oisst.py` and `sources/era5.py` as the reference implementations).

## Sister repo

`~/src/globe-viz` — Astro/Solid/Three.js frontend. It consumes our S3 outputs at runtime; we never push to it directly. Both repos share one approved expansion plan at `/Users/garyo/.claude/plans/tranquil-imagining-seal.md` (multi-source: ERA5 + MODIS LST; regional aggregates including Niño 3.4).

## Deploy / runtime

GitHub Actions workflow `.github/workflows/make-images.yml` runs nightly at 13:15 UTC (just after NOAA's 9 AM EDT update). It prunes the last 90 days from cache (catches reanalyses), regenerates graphs/maps/textures, exports time-series JSON, and uploads everything to S3 (`climate-change-assets/sea-surface-temp/`). AWS creds in `.env` (gitignored).

`upload-to-s3.py` regenerates `index.json` from the S3 bucket listing — the bucket is the source of truth for available dates and regions, not the local repo.

## Important behaviors that aren't obvious from the code

- **`data-cache.json.gz` is persisted to S3, not committed back from CI.** The committed cache is the authority for *historical* dates (curated by deliberate backfill commits) and is frozen well in the past — it can lag months behind reality. *Recent* dates are never committed; instead each CI run restores `s3://climate-change-assets/sea-surface-temp/data-cache.json.gz`, merges it onto the committed one (`scripts/merge_cache.py` — committed wins on shared keys, S3 fills the recent tail), then pushes the result back at the end. This is what makes a successfully-collected recent day durable: before it, recent ERA5 was rebuilt from scratch each run off the ephemeral `era5-archive/`, so any date an upstream outage missed while it sat in the fetch window was lost forever once it aged out (that's how the May–Jun 2026 Niño-3.4 gap became permanent). The daily backfill now retries still-missing dates over a **90-day** window, so a gap self-heals whenever the source republishes. For testing aggregation changes, refresh locally first (the committed cache lags):

  ```sh
  uv run prune-cache.py --inplace --days 90
  uv run pipeline.py --source oisst --mode graph --dataset sst --out /tmp/x.png
  ```

  Note `prune-cache.py` defaults to `--sources oisst`, so it only prunes OISST entries; ERA5 entries are kept regardless of age (they're refilled from the S3 cache + backfill, not refetched wholesale each run).

- **The cache is gzipped, and all I/O goes through `cache_io.py`.** `data-cache.json.gz` is the committed file; nothing opens it directly. `cache_io.load_cache`/`save_cache` pick compression from the `.gz` suffix (a plain `.json` path still works, so a pre-gzip cache file stays readable), round values to 6 decimals, and write atomically via a `.part` temp file. Two things not to undo: compression, because at 58 MB plain the committed file drew GitHub's large-file warning and was heading for the 100 MB hard limit (gzip takes it to ~7 MB); and rounding, because these are area-weighted means of float32 grids — OISST's own data is `int16 * 0.01`, so 17 significant digits stored pure noise, cost ~20% of the file, and inflated every `timeseries/*.json` the browser downloads by ~30%. When adding a tool that touches the cache, import `cache_io` rather than calling `json.load` — the temp file's suffix is `.part`, so anything deriving compression from the path it is *writing* will silently produce a plain file under a `.gz` name.

- **Cache key** format: `YYYY-MM-DD-{source}-{dataset}-{region}` → float (cosine-lat-weighted average over the named region). Source is `oisst` or `era5`; dataset is one of the source's exposed names (`sst`/`anom` for OISST, `sst`/`t2m` for ERA5); region is one of `regions.REGIONS`. `pipeline.py` and `scripts/aggregate_archive.py` both write this same shape. The one exception is the reserved `{date}-{source}-preliminary-flag` key described below, which is a provenance marker rather than a measurement.

- **OISST preliminary data is tagged, not hidden.** NOAA publishes `oisst-avhrr-v02r01.YYYYMMDD_preliminary.nc` for ~2 weeks and then replaces it with the final file; values shift by ~0.01 °C. `OisstSource.is_preliminary()` detects this from the NetCDF **`id` attribute** (which holds the original filename) — *not* from the local filename, because `backfill_oisst.py` saves both URLs under the final name, and not from `title`, which NOAA stores truncated mid-word ("… Version 2.1 - Inter"). When a fetched file is provisional, the pipeline writes a `{date}-{source}-preliminary-flag` cache key (value `1.0`; absence means final). That key is deliberately shaped like a normal cache key with a reserved dataset/region so it rides the existing machinery: `prune-cache.py` still reads `oisst` out of it, so the nightly 90-day prune drops it and the re-fetch re-writes it only if the day is *still* provisional — the flags self-heal with no merge semantics to design. `export_timeseries.py` pulls those keys out before grouping (so no bogus `flag` region appears) and emits a per-source `preliminary_from` date in each region JSON; the frontend draws everything from that date on as a dotted line. Two consequences worth knowing:
  - `scripts/newly_finalized.py` diffs the flags across a run so the texture step re-renders exactly the days that just finalized. Without it every published texture keeps the provisional imagery it was first rendered from, since the render loop only covers `days_ago` 2..8 but finalization lands at ~D+14. It needs a *previous* run's snapshot to diff against, so it re-renders nothing on its first run — to backfill days rendered before it existed, trigger the workflow manually with the `extra_texture_days` input set to `9 10 11 12 13 14 15 16`. That input is filtered down to bare integers, so it can't inject shell.
  - `backfill_oisst.py --force` is the only way to upgrade an archived day that was downloaded while still preliminary — both URLs write the same final filename, so the normal skip-if-exists check would never revisit it.

- **ERA5 archive** lives under `./era5-archive/YYYY/era5-YYYYMMDD.nc` (gitignored). Files are pre-resampled to the OISST 720×1440 grid + zlib-compressed at fetch time (~3 MB/day). `scripts/backfill_era5.py` does an overnight bulk fetch; the daily cron tops it up via `pipeline.py --source era5 --mode texture`. Requires `~/.cdsapirc` locally and the `CDS_API_KEY` GH secret.

- **`export_timeseries.py`** writes one JSON per region into `maps/timeseries/`. `upload-to-s3.py` walks subdirs recursively, so they land at `sea-surface-temp/timeseries/{region}.json` on S3. Don't flatten that layout — `index.json`'s `timeseries.regions` field is regenerated from the directory structure.

- **Globe textures start 2024-01-01; graphs use the full record.** Map textures (equirect WebP) are only published from **2024-01-01** onward — older history would bloat users' browsers and S3 for little benefit. `generate_era5_textures_batch.py` defaults `--start 2024-01-01`; **time-series JSON (graphs) is never floored** — it covers the whole archive (ERA5 to 1982, GFS to 2021-03-23). A bulk GFS/ERA5 backfill renders earlier textures as a side effect of region aggregation (one `get_data_array` feeds both the texture and the cache), so before uploading, keep `<2024` texture files out of `./maps` while keeping every date in `data-cache.json`/`timeseries/`. `upload-to-s3.py` publishes whatever is in `./maps`, so the date floor is enforced by what's present, not by the uploader.

## Conventions

- `uv` for everything Python (per global CLAUDE.md), type hints, ruff.
- A pre-commit hook (`.githooks/pre-commit`) runs `ruff check` + `ruff format --check` on staged Python files. One-time setup after cloning: `git config core.hooksPath .githooks`.
- "Today" means UTC (`pipeline.utc_today()` / `datetime.now(UTC).date()`) — data dates follow the sources' UTC calendar and CI runs in UTC; don't reintroduce naive `date.today()`.
- Don't add a "refresh cache" commit — the cache file changes whenever you run the pipeline locally; only commit it when the change is intentional (e.g. a backfill).
