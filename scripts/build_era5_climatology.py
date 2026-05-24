#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Build a daily ERA5 SST climatology over the 1971-2000 baseline.

Streams ERA5 daily-mean SST month-at-a-time from CDS, accumulates per-DOY
sums on the OISST 720×1440 grid, applies a 7-day centered rolling mean
along day-of-year, and writes a single NetCDF.

**Why 7-day smoothing?** OISST's published anomaly is built against a hybrid
climatology — weekly OISST v2 (1982-2000) blended with an in-situ
climatology (1971-2000), then interpolated to daily values (NOAA CDR ATBD
CDRP-ATBD-0303 §3.3.2). The weekly aggregation step is OISST's de-facto
temporal smoother. We approximate it on the daily ERA5 mean with a 7-day
centered rolling mean. The result has effectively-comparable temporal
character so anomalies overlay sensibly against OISST anom.

**Caveat:** 1971-1978 is ERA5 back-extension data — different observation
density than 1979+ (pre-satellite SST). The bias is small for the
climatological mean (averaging across many years dilutes it) and matching
the OISST 1971-2000 baseline is more important for visual comparability
than avoiding the back-extension.

Output:
    NetCDF, dims (doy=366, latitude=720, longitude=1440), var `sst_climatology`
    in °C, zlib level 4. ~100-150 MB compressed.

Wall time: ~1.5-3 hours, dominated by CDS queue. Each monthly request
fetches all days of that month in one shot — 360 requests total vs ~11,000
single-day requests.

Memory: ~3 GB peak (sum accumulator at float64 + count accumulator).

Usage:
    uv run scripts/build_era5_climatology.py
    # then upload:
    aws s3 cp ./era5-archive/climatology/era5-climatology-1971-2000.nc \\
        s3://climate-change-assets/sea-surface-temp/climatology/
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Sibling sources package import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sources.era5 import _resample_to_oisst_grid  # noqa: E402

if TYPE_CHECKING:
    import xarray as xr


GRID_SHAPE = (720, 1440)
DOY_LEAP = 366


def fetch_month(year: int, month: int) -> "xr.Dataset":
    """Fetch one month of ERA5 daily-mean SST in one CDS request.

    Returns the resampled Dataset (OISST grid). Variable name is ``sst``;
    units are Kelvin per CDS convention — convert downstream.
    """
    import cdsapi
    import xarray as xr

    n_days = calendar.monthrange(year, month)[1]
    request = {
        "product_type": "reanalysis",
        "variable": ["sea_surface_temperature"],
        "year": f"{year:04}",
        "month": f"{month:02}",
        "day": [f"{d:02}" for d in range(1, n_days + 1)],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
    }

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # CDS returns a zip bundle when multiple variables are requested
        # (one NetCDF per variable) but a bare NetCDF when only one is
        # requested. We only want SST → write to a generic filename and
        # branch on the response shape.
        response_path = tmpdir / "response.bin"

        client = cdsapi.Client()
        client.retrieve(
            "derived-era5-single-levels-daily-statistics",
            request,
            str(response_path),
        )

        if zipfile.is_zipfile(response_path):
            with zipfile.ZipFile(response_path) as zf:
                zf.extractall(tmpdir)
            nc_files = sorted(p for p in tmpdir.glob("*.nc") if p != response_path)
            if not nc_files:
                raise RuntimeError(
                    f"CDS zip for {year}-{month:02} had no NetCDFs: "
                    f"{[n.name for n in tmpdir.iterdir()]}"
                )
            parts = []
            for f in nc_files:
                with xr.open_dataset(f) as ds:
                    parts.append(ds.load())
            merged = xr.merge(parts, compat="override")
        else:
            with xr.open_dataset(response_path) as ds:
                merged = ds.load()

        resampled = _resample_to_oisst_grid(merged).load()
        return resampled


def leap_year_doy(date: datetime.date) -> int:
    """Day-of-year in a leap-year calendar (1..366).

    Maps every non-leap date to the DOY it would occupy in a leap year:
    Jan/Feb (months 1-2) keep their natural DOY; Mar onward shifts by +1.
    Feb 29 → 60 (only contributed by actual leap years).
    """
    natural = date.timetuple().tm_yday
    if calendar.isleap(date.year) or date.month < 3:
        return natural
    return natural + 1


def smooth_doy(daily: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along the DOY axis (axis 0), wrapping at year boundary.

    NaN-aware: pixels that are NaN in some neighbors average over the valid
    ones; fully-masked positions stay NaN.
    """
    if window % 2 != 1:
        raise ValueError("window must be odd")
    assert daily.shape[0] == DOY_LEAP

    pad = window // 2
    # Wrap-pad so DOY 1's window includes the last `pad` days of the prior year.
    padded = np.concatenate([daily[-pad:], daily, daily[:pad]], axis=0)

    out = np.empty_like(daily)
    for d in range(DOY_LEAP):
        chunk = padded[d : d + window]
        valid = ~np.isnan(chunk)
        n = valid.sum(axis=0)
        s = np.where(valid, chunk, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[d] = np.where(n > 0, s / n, np.nan)
    return out


def enumerate_months(start: datetime.date, end: datetime.date) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples covering [start, end]."""
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def accumulate_month(
    ds: "xr.Dataset",
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
) -> int:
    """Fold one month's daily SST into the running per-DOY sums. Returns days added."""
    sst = ds["sst"]
    # CDS time dim is 'valid_time'; tolerate 'time' for safety.
    tdim = "valid_time" if "valid_time" in sst.dims else "time"
    n = 0
    for t in range(sst.sizes[tdim]):
        slab = sst.isel({tdim: t})
        date = _as_date(slab[tdim].values)
        doy = leap_year_doy(date)
        arr_c = np.asarray(slab.values, dtype=np.float64) - 273.15
        valid = ~np.isnan(arr_c)
        sum_grid[doy - 1][valid] += arr_c[valid]
        count_grid[doy - 1][valid] += 1
        n += 1
    return n


def _as_date(np_dt) -> datetime.date:
    """numpy.datetime64 (any precision) → datetime.date."""
    ts = np.datetime64(np_dt, "D").astype("O")
    return datetime.date(ts.year, ts.month, ts.day)


def save_climatology(
    path: Path,
    clim: np.ndarray,
    start: datetime.date,
    end: datetime.date,
    smoothing_window: int,
) -> None:
    """Write the climatology to NetCDF with zlib compression."""
    import xarray as xr

    # Reuse the canonical OISST cell centers as coords (same grid as
    # everything else in the project).
    lats = np.arange(-89.875, 90.0, 0.25)
    lons = np.arange(0.125, 360.0, 0.25)

    da = xr.DataArray(
        clim.astype(np.float32),
        dims=("doy", "latitude", "longitude"),
        coords={
            "doy": np.arange(1, DOY_LEAP + 1, dtype=np.int16),
            "latitude": lats.astype(np.float32),
            "longitude": lons.astype(np.float32),
        },
        attrs={
            "long_name": "ERA5 sea-surface temperature climatology (daily mean)",
            "units": "degC",
            "baseline_start": start.isoformat(),
            "baseline_end": end.isoformat(),
            "smoothing": f"{smoothing_window}-day centered rolling mean along DOY (wrapped)",
            "source": "ECMWF ERA5 (CDS derived-era5-single-levels-daily-statistics, daily_mean)",
            "grid": "OISST 0.25° (720x1440, lat ascending -89.875..89.875, lon 0.125..359.875)",
            "comment": (
                "Approximates OISST's weekly-aggregate climatology methodology "
                "(NOAA CDR ATBD CDRP-ATBD-0303 §3.3.2) by smoothing the raw "
                "30-year daily mean with a 7-day centered rolling window. "
                "1971-1978 uses ERA5 back-extension."
            ),
        },
        name="sst_climatology",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[Any, dict[str, Any]] = {
        "sst_climatology": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }
    da.to_netcdf(path, encoding=encoding)


def build(
    start_date: datetime.date,
    end_date: datetime.date,
    out_path: Path,
    workers: int,
    smoothing_window: int,
) -> int:
    months = enumerate_months(start_date, end_date)
    print(f"Building ERA5 SST climatology {start_date}..{end_date} ({len(months)} months)")
    print(f"  Output: {out_path}")
    print(f"  Workers: {workers}, smoothing: {smoothing_window}-day rolling mean")
    print()

    sum_grid = np.zeros((DOY_LEAP, *GRID_SHAPE), dtype=np.float64)
    count_grid = np.zeros((DOY_LEAP, *GRID_SHAPE), dtype=np.int32)

    started = time.time()
    failed: list[tuple[int, int]] = []
    n_days_processed = 0

    def task(ym: tuple[int, int]):
        return ym, fetch_month(*ym)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, ym): ym for ym in months}
        for i, fut in enumerate(as_completed(futures), 1):
            ym = futures[fut]
            try:
                _, ds = fut.result()
            except Exception as e:  # noqa: BLE001 — surface CDS errors with context
                failed.append(ym)
                print(f"  ❌ {ym[0]}-{ym[1]:02} fetch failed: {type(e).__name__}: {e}")
                continue
            try:
                added = accumulate_month(ds, sum_grid, count_grid)
                n_days_processed += added
            finally:
                ds.close()
            elapsed = time.time() - started
            rate = i / max(elapsed, 1)
            eta = (len(months) - i) / max(rate, 0.001)
            print(
                f"[{i}/{len(months)}] {ym[0]}-{ym[1]:02} +{added}d "
                f"(total {n_days_processed}d, {rate*60:.1f} mo/min, ETA {eta/60:.1f}m)"
            )

    if failed:
        print()
        print(f"⚠️  {len(failed)} months failed: {failed}")
        if len(failed) > len(months) // 20:
            print("Too many failures (>5%). Aborting before write.")
            return 1
        print("Continuing with partial coverage; gaps will show as low count_grid pixels.")

    print()
    print("Computing per-pixel mean...")
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(count_grid > 0, sum_grid / count_grid, np.nan).astype(np.float32)

    # Diagnostic: how well-covered is each DOY?
    cov = (count_grid > 0).any(axis=(1, 2)).sum()
    print(f"  DOYs with any data: {cov}/{DOY_LEAP}")
    if cov < DOY_LEAP:
        # Find which DOYs are empty
        empty = [d + 1 for d in range(DOY_LEAP) if not (count_grid[d] > 0).any()]
        print(f"  Empty DOYs: {empty}")

    # Diagnostic: count distribution
    nonzero = count_grid[count_grid > 0]
    if nonzero.size:
        print(
            f"  Sample count per pixel-DOY: median={np.median(nonzero):.0f} "
            f"min={nonzero.min()} max={nonzero.max()}"
        )

    print(f"Applying {smoothing_window}-day rolling-mean smoothing...")
    clim_smoothed = smooth_doy(clim, smoothing_window)

    print(f"Writing {out_path}")
    save_climatology(out_path, clim_smoothed, start_date, end_date, smoothing_window)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✓ Wrote {out_path} ({size_mb:.1f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.date(1971, 1, 1),
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.date(2000, 12, 31),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./era5-archive/climatology/era5-climatology-1971-2000.nc"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel CDS requests (CDS caps each user around 6)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=7,
        help="Odd window size for the DOY rolling mean (7 ≈ OISST's weekly aggregate)",
    )
    args = parser.parse_args(argv)

    if args.end < args.start:
        print("❌ --end must be >= --start", file=sys.stderr)
        return 2

    return build(
        start_date=args.start,
        end_date=args.end,
        out_path=args.out,
        workers=args.workers,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
