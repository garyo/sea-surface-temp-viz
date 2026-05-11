#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Backfill ERA5 daily-mean SST + 2 m air temp from ARCO-ERA5 on Google Cloud.

The Copernicus CDS API throttles per-user concurrency to ~2 simultaneous
requests regardless of how many workers we submit, capping a multi-year
backfill at ~0.03 days/sec (~20 h for a 5-year range). ARCO-ERA5 — the
analysis-ready, cloud-optimized public mirror at
``gs://gcp-public-data-arco-era5/`` — has no queue, no auth, and chunks
each hourly field as one ~4 MB Zarr blob. Pulling 24 hourly fields and
averaging them locally produces the same daily mean two orders of
magnitude faster.

The output file format and grid match :mod:`sources.era5` exactly, so
``aggregate_archive.py`` and the rest of the pipeline are unaware of the
fetch path. ERA5T (the preliminary stream, ~1-week latency) extends the
stable archive to within a few days of present; the daily CDS cron keeps
filling the most recent slice.

Usage:
    uv run scripts/backfill_era5_arco.py [--start 2021-01-01] [--workers 8]
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sources.era5 import Era5Source, _resample_to_oisst_grid  # noqa: E402

ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ARCO variable names → output names we already write from the CDS path.
# Aligning these means downstream code (sources/era5.py, aggregate_archive,
# pipeline.py) doesn't need to know which fetch route produced the file.
ARCO_VARS = {
    "sea_surface_temperature": "sst",
    "2m_temperature": "t2m",
}

DEFAULT_START = datetime.date(2021, 1, 1)


def open_arco() -> xr.Dataset:
    """Open the public ARCO-ERA5 zarr with anonymous GCS access."""
    return xr.open_zarr(
        ARCO_ZARR,
        chunks=None,
        storage_options={"token": "anon"},
    )[list(ARCO_VARS.keys())]


def build_daily_mean(ds: xr.Dataset, d: datetime.date) -> xr.Dataset:
    """Average the 24 hourly fields for UTC day ``d`` into one daily field.

    Returns an xr.Dataset matching the shape produced by the CDS daily-stats
    path: ``(valid_time=1, latitude, longitude)`` per variable, renamed to
    the short names (``sst``, ``t2m``) the pipeline already consumes.
    """
    start = datetime.datetime(d.year, d.month, d.day, 0)
    end = datetime.datetime(d.year, d.month, d.day, 23)
    day = ds.sel(time=slice(start, end))
    n = day.sizes["time"]
    if n != 24:
        raise RuntimeError(f"Expected 24 hourly steps for {d}, got {n}")
    mean = day.mean(dim="time", keep_attrs=True).rename(ARCO_VARS)
    return mean.expand_dims(valid_time=[np.datetime64(d, "ns")])


def write_one(
    d: datetime.date, archive_root: Path, ds: xr.Dataset
) -> tuple[datetime.date, str]:
    """Returns (date, status) where status is 'ok', 'skip', or 'fail: <reason>'."""
    out = Era5Source.archive_path(archive_root, d)
    if out.exists() and out.stat().st_size > 0:
        return (d, "skip")
    try:
        daily = build_daily_mean(ds, d).load()
        resampled = _resample_to_oisst_grid(daily).load()
        out.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in resampled.data_vars
        }
        resampled.to_netcdf(out, encoding=encoding)
        return (d, "ok")
    except Exception as e:  # noqa: BLE001 — surface for triage like the CDS variant
        return (d, f"fail: {type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root", type=Path, default=Era5Source.archive_root
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.date.today() - datetime.timedelta(days=7),
        help="Inclusive end date (default: 7 days before today, past ERA5T latency)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent GCS readers (each fetches 24 hourly chunks per day)",
    )
    args = parser.parse_args(argv)

    args.archive_root.mkdir(parents=True, exist_ok=True)

    print(f"Backfilling ERA5 from ARCO/GCS: {args.start} → {args.end}")
    print(f"Archive root: {args.archive_root.resolve()}")
    print(f"Workers: {args.workers}")
    print()

    print("Opening ARCO zarr...")
    ds = open_arco()
    stable_stop = ds.attrs.get("valid_time_stop")
    era5t_stop = ds.attrs.get("valid_time_stop_era5t")
    print(f"  stable ERA5 through:    {stable_stop}")
    print(f"  ERA5T preliminary thru: {era5t_stop}")

    # Clamp end to ERA5T cutoff so we don't request NaN-filled future slots.
    if era5t_stop:
        cutoff = datetime.datetime.strptime(era5t_stop, "%Y-%m-%d").date()
        if args.end > cutoff:
            print(f"  ⚠  clamping end {args.end} → {cutoff} (ERA5T cutoff)")
            args.end = cutoff

    one = datetime.timedelta(days=1)
    dates: list[datetime.date] = []
    d = args.start
    while d <= args.end:
        dates.append(d)
        d += one
    print(f"  {len(dates)} days to process")
    print()

    started = time.time()
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(write_one, d, args.archive_root, ds) for d in dates
        ]
        for i, f in enumerate(as_completed(futures), 1):
            d, status = f.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  ❌ {d}: {status}")
            if i % 20 == 0 or i == len(futures):
                elapsed = time.time() - started
                rate = i / max(elapsed, 1)
                eta = (len(futures) - i) / max(rate, 0.001)
                print(
                    f"[{i}/{len(futures)}] ok={ok} skip={skip} fail={fail} "
                    f"({rate:.2f}/s, ETA {eta/60:.1f}m)"
                )

    print()
    print(f"Done. Wrote: {ok}, Already had: {skip}, Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
