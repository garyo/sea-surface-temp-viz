#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Build the Δ fields and regional offsets for alternative anomaly baselines.

See ``baselines.py`` for what Δ means and how it's applied. This script writes:

    baselines/fields/delta-{field}-{baseline}.nc   (month=12, 720×1440, °C)
    baselines/fields/valid-{oisst,era5}-sst.npy    (valid-cell masks)
    baselines/offsets.json                         (committed; region means)
    {publish dir}/baselines/delta-*.webp, manifest.json   (the globe's copy)

``fields/`` is gitignored (the Δ grids are also uploaded to S3 for the globe);
``offsets.json`` is small and committed, so the nightly export needs neither.
Downloads are cached under ``baselines/raw/`` (~5 GB in all), so re-runs are
fast.

**1991–2020.** ERA5 and GFS anomalies are measured against ERA5, so their Δ
comes from ERA5 monthly means (CDS ``reanalysis-era5-single-levels-monthly-
means``, a decade per request). OISST's comes from NOAA PSL's daily OISST
climatologies for 1971–2000 (the one its anomalies use) and 1991–2020.

**Pre-industrial (1850–1900).** No reanalysis or satellite record reaches back
that far, so each source's Δ is its own 1971–2000 → 1991–2020 Δ (above) plus
the 1991–2020 → 1850–1900 difference from long-record gridded reconstructions
(see PREINDUSTRIAL_ANCHOR for why 1991–2020), averaged across datasets because
their spread is the dominant uncertainty:

- 2 m air temperature: Berkeley Earth (air temperature over sea ice), HadCRUT5
  analysis, and NOAAGlobalTemp v6.1 — the trio Copernicus averages for its
  global pre-industrial offset. They blend SST over the ocean; IPCC AR6 (Cross-
  Chapter Box 2.3) puts the long-term difference from true air temperature at
  ≤10%.
- SST: ERSSTv6 and COBE-SST2, both infilled back to 1850. HadSST4 (not
  infilled) is printed as a cross-check but not averaged in.

Per dataset and calendar month, a cell's 1850–1900 mean counts only if at least
70% of those years have data there. On a common 1° grid, each dataset's gaps
are filled from the most complete one (shifted by their mean difference) and
the datasets are averaged; cells none of them cover are filled from their 5°
latitude band (which keeps polar amplification, unlike a global constant),
then the field is
smoothed with a ~500 km Gaussian — finer structure before 1900 is statistical
reconstruction, not observation.

**All fields** are reduced to a mean plus two annual harmonics across the 12
months (Copernicus fits one harmonic to its global offsets), gap-filled from the
nearest valid cell so every cell any source reports is covered, and written on
the OISST grid.

Usage:
    uv run scripts/build_baseline_deltas.py                # everything
    uv run scripts/build_baseline_deltas.py --baseline preindustrial  # after 1991-2020
    uv run scripts/build_baseline_deltas.py --offsets-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import baselines
from sources.era5 import _OISST_LATS, _OISST_LONS, _resample_to_oisst_grid
from textures import Rg16, save_lossless_webp

if TYPE_CHECKING:
    import xarray as xr

RAW_DIR = baselines.BASELINE_DIR / "raw"

# CDS variable name → the short name it comes back as in the NetCDF.
ERA5_VARIABLES = {"sst": "sea_surface_temperature", "t2m": "2m_temperature"}
KELVIN = 273.15
DECADE = 10

PSL = "https://downloads.psl.noaa.gov/Datasets"
OISST_LTM_URL = PSL + "/noaa.oisst.v2.highres/sst.day.mean.ltm.{y0}-{y1}.nc"

# One final (non-preliminary) OISST day, read only for its land mask, which is
# fixed across the record.
OISST_DAY_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr/202401/oisst-avhrr-v02r01.20240101.nc"
)

# Fraction of a period's years that must have data for a cell's monthly mean.
MIN_COVERAGE = 0.7
# Intermediate grid the reconstructions are combined on, and the latitude
# bands used to fill cells none of them cover.
COMBINE_RES = 1.0
FILL_BAND_DEG = 5.0
SMOOTHING_KM = 500.0
KM_PER_DEG = 111.2
SEASONAL_HARMONICS = 2


@dataclass(frozen=True)
class Reconstruction:
    name: str
    url: str  # a trailing "/" means: the newest .nc in that directory listing
    var: str
    field: str  # which Δ field it feeds: "sst" or "t2m"
    averaged: bool = True  # False = printed as a cross-check only


RECONSTRUCTIONS = (
    Reconstruction("ERSSTv6", f"{PSL}/noaa.ersst.v6/sst.mnmean.nc", "sst", "sst"),
    Reconstruction("COBE-SST2", f"{PSL}/COBE2/sst.mon.mean.nc", "sst", "sst"),
    Reconstruction(
        "HadSST4",
        "https://www.metoffice.gov.uk/hadobs/hadsst4/data/data/HadSST.4.2.0.0_median.nc",
        "tos",
        "sst",
        averaged=False,
    ),
    Reconstruction(
        "Berkeley Earth",
        "https://storage.googleapis.com/berkeley-earth-temperature/Global/Gridded/"
        "Land_and_Ocean_LatLong1.nc",
        "temperature",
        "t2m",
    ),
    Reconstruction(
        "HadCRUT5",
        "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.2.0.0/analysis/"
        "HadCRUT.5.2.0.0.analysis.anomalies.ensemble_mean.nc",
        "tas_mean",
        "t2m",
    ),
    Reconstruction(
        "NOAAGlobalTemp",
        "https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6.1/"
        "access/gridded/",
        "anom",
        "t2m",
    ),
)


# ---------------------------------------------------------------- downloads


def download(url: str) -> Path:
    """Fetch ``url`` into RAW_DIR once; a directory URL picks its newest .nc."""
    if url.endswith("/"):
        with urllib.request.urlopen(url, timeout=60) as resp:
            listing = resp.read().decode()
        url += max(re.findall(r'href="([^"/]+\.nc)"', listing))
    out = RAW_DIR / url.rsplit("/", 1)[1]
    if not out.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        print(f"↓ {url}")
        urllib.request.urlretrieve(url, part)
        part.rename(out)
    return out


def fetch_era5_monthly(variable: str, first_year: int, last_year: int) -> Path:
    """Download (or reuse) one CDS request of ERA5 monthly means."""
    out = RAW_DIR / f"era5-moda-{variable}-{first_year}-{last_year}.nc"
    if out.exists():
        return out
    import cdsapi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [ERA5_VARIABLES[variable]],
        "year": [str(y) for y in range(first_year, last_year + 1)],
        "month": [f"{m:02}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    with tempfile.TemporaryDirectory(dir=RAW_DIR) as tmp:
        response = Path(tmp) / "response"
        cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels-monthly-means", request, str(response)
        )
        if zipfile.is_zipfile(response):
            with zipfile.ZipFile(response) as zf:
                (member,) = [n for n in zf.namelist() if n.endswith(".nc")]
                zf.extract(member, tmp)
                response = Path(tmp) / member
        response.rename(out)
    print(f"✓ {out}")
    return out


# ------------------------------------------------------------ field helpers


def fill_nearest(field: np.ndarray) -> np.ndarray:
    """Replace NaNs in each 2D slice with the nearest finite cell's value."""
    from scipy import ndimage

    out = field.copy()
    for month in out:
        missing = np.isnan(month)
        if missing.any():
            _, (ii, jj) = ndimage.distance_transform_edt(missing, return_indices=True)
            month[missing] = month[ii[missing], jj[missing]]
    return out


def smooth_seasonal(delta: np.ndarray) -> np.ndarray:
    """Least-squares fit of a mean + SEASONAL_HARMONICS harmonics to 12 months."""
    spectrum = np.fft.rfft(delta, axis=0)
    spectrum[SEASONAL_HARMONICS + 1 :] = 0
    return np.fft.irfft(spectrum, n=12, axis=0).astype(np.float32)


def global_mean(field: np.ndarray, lats: np.ndarray) -> float:
    """Cos-lat-weighted mean of the annual-mean field over cells with data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
        annual = np.nanmean(field, axis=0) if field.ndim == 3 else field
    w = np.broadcast_to(np.cos(np.radians(lats))[:, None], annual.shape)
    ok = np.isfinite(annual)
    return float(np.average(annual[ok], weights=w[ok]))


def to_grid(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    new_lats: np.ndarray,
    new_lons: np.ndarray,
) -> np.ndarray:
    """Bilinearly regrid (…, lat, lon) data, wrapping in longitude.

    ``lats`` must ascend and ``lons`` ascend within [0, 360).
    """
    import xarray as xr

    lead = [f"d{i}" for i in range(field.ndim - 2)]
    da = xr.DataArray(
        field, dims=(*lead, "lat", "lon"), coords={"lat": lats, "lon": lons}
    )
    padded = xr.concat(
        [
            da.isel(lon=[-1]).assign_coords(lon=lons[-1:] - 360),
            da,
            da.isel(lon=[0]).assign_coords(lon=lons[:1] + 360),
        ],
        dim="lon",
    )
    return padded.interp(lat=new_lats, lon=new_lons).values.astype(np.float32)


def write_delta(baseline_id: str, field: str, delta: np.ndarray) -> None:
    import xarray as xr

    delta = smooth_seasonal(fill_nearest(delta))
    path = baselines.delta_path(baseline_id, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"delta": (("month", "latitude", "longitude"), delta)},
        coords={
            "month": np.arange(1, 13),
            "latitude": _OISST_LATS,
            "longitude": _OISST_LONS,
        },
        attrs={
            "description": (
                f"{field} mean over {baselines.REFERENCE_YEARS} minus mean over "
                f"{baselines.BASELINES[baseline_id].years}, per calendar month; "
                "add to a 1971–2000 anomaly to re-baseline it"
            ),
            "units": "degC",
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    ds.to_netcdf(path, encoding={"delta": {"zlib": True, "complevel": 4}})
    print(f"✓ {path} (global mean {global_mean(delta, _OISST_LATS):+.3f})")


# ------------------------------------------------------------ 1991–2020


def era5_period_means(
    variable: str, periods: dict[str, tuple[int, int]]
) -> dict[str, xr.DataArray]:
    """Per-calendar-month means (°C) over each period, on ERA5's native grid.

    Accumulates a file (decade) at a time, so peak memory is one decade of
    fields plus the float64 sums.
    """
    import xarray as xr

    first = min(y0 for y0, _ in periods.values())
    last = max(y1 for _, y1 in periods.values())
    chunks = [(y, min(y + DECADE - 1, last)) for y in range(first, last + 1, DECADE)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda c: fetch_era5_monthly(variable, *c), chunks))

    sums: dict[str, np.ndarray] = {}
    counts = {name: np.zeros(12, dtype=np.int64) for name in periods}
    for path in paths:
        with xr.open_dataset(path) as ds:
            da = ds[variable]
            coords = {"latitude": da["latitude"], "longitude": da["longitude"]}
            values = da.values.astype(np.float64) - KELVIN
            years = da["valid_time"].dt.year.values
            months = da["valid_time"].dt.month.values
        for name, (y0, y1) in periods.items():
            total = sums.setdefault(name, np.zeros((12, *values.shape[1:])))
            for i in np.flatnonzero((years >= y0) & (years <= y1)):
                total[months[i] - 1] += values[i]
                counts[name][months[i] - 1] += 1

    expected = {name: y1 - y0 + 1 for name, (y0, y1) in periods.items()}
    for name, n in counts.items():
        if (n != expected[name]).any():
            raise RuntimeError(
                f"{variable} {name}: month counts {n}, expected {expected[name]}"
            )
    return {
        name: xr.DataArray(
            (sums[name] / counts[name][:, None, None]).astype(np.float32),
            dims=("month", "latitude", "longitude"),
            coords={"month": np.arange(1, 13), **coords},
        )
        for name in periods
    }


def build_era5_1991_2020() -> None:
    target = baselines.BASELINES["1991-2020"].years
    baselines.FIELD_DIR.mkdir(parents=True, exist_ok=True)
    for variable in ERA5_VARIABLES:
        means = era5_period_means(
            variable, {"ref": baselines.REFERENCE_YEARS, "target": target}
        )
        native = (means["ref"] - means["target"]).rename({"month": "valid_time"})
        delta = _resample_to_oisst_grid(native.to_dataset(name="delta"))["delta"]
        delta = np.asarray(delta.values, dtype=np.float32)
        if variable == "sst":
            # Same interpolation as the daily ERA5 textures, so the same cells
            # come out NaN as in the data the offsets are added to.
            np.save(
                baselines.FIELD_DIR / "valid-era5-sst.npy",
                ~np.isnan(delta).all(axis=0),
            )
        write_delta("1991-2020", variable, delta)


def build_oisst_1991_2020() -> None:
    """Monthly means of NOAA's daily OISST climatology difference."""
    import xarray as xr

    ref, target = (
        download(OISST_LTM_URL.format(y0=y0, y1=y1))
        for y0, y1 in (
            baselines.REFERENCE_YEARS,
            baselines.BASELINES["1991-2020"].years,
        )
    )
    with (
        xr.open_dataset(ref, decode_times=False) as a,
        xr.open_dataset(target, decode_times=False) as b,
    ):
        n_days = a.sizes["time"]
        if b.sizes["time"] != n_days or a["sst"].shape[1:] != (720, 1440):
            raise RuntimeError(f"Unexpected OISST climatology shapes: {a}, {b}")
        # Climatology files carry a nominal year; recover months by position.
        year = 2000 if n_days == 366 else 2001
        months = np.array(
            [
                (datetime.date(year, 1, 1) + datetime.timedelta(days=d)).month
                for d in range(n_days)
            ]
        )
        delta = np.stack(
            [
                np.mean(
                    a["sst"].isel(time=np.flatnonzero(months == m)).values
                    - b["sst"].isel(time=np.flatnonzero(months == m)).values,
                    axis=0,
                )
                for m in range(1, 13)
            ]
        )
        lats = a["lat"].values
    if lats[0] > lats[-1]:
        delta = delta[:, ::-1]
    write_delta("1991-2020", "oisst", delta)


def build_1991_2020() -> None:
    build_era5_1991_2020()
    build_oisst_1991_2020()


# -------------------------------------------------------- pre-industrial


def open_monthly(path: Path, var: str) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """A reconstruction as (time, lat↑, lon 0–360↑), plus each step's year, month."""
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    da = ds[var].squeeze(drop=True)
    da = da.rename(
        {k: v for k, v in (("lat", "latitude"), ("lon", "longitude")) if k in da.dims}
    )
    t = ds["time"]
    if str(t.attrs.get("units", "")).startswith("year"):
        # Berkeley Earth: decimal years at mid-month (1850.0417 = Jan 1850).
        years = np.floor(t.values).astype(int)
        months = np.floor((t.values - years) * 12).astype(int) + 1
    else:
        dates = xr.decode_cf(ds[["time"]])["time"].dt
        years, months = dates.year.values, dates.month.values
    da = da.assign_coords(longitude=da["longitude"] % 360).sortby(
        ["latitude", "longitude"]
    )
    return da, years, months


def period_monthly_mean(
    da: xr.DataArray, years: np.ndarray, months: np.ndarray, period: tuple[int, int]
) -> np.ndarray:
    """(12, lat, lon) means; NaN where under MIN_COVERAGE of the years have data."""
    y0, y1 = period
    need = MIN_COVERAGE * (y1 - y0 + 1)
    out = []
    for m in range(1, 13):
        values = da.isel(
            time=np.flatnonzero((years >= y0) & (years <= y1) & (months == m))
        ).values
        n = np.isfinite(values).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
            mean = np.nanmean(values, axis=0)
        out.append(np.where(n >= need, mean, np.nan))
    return np.stack(out).astype(np.float32)


def fill_from_bands(
    field: np.ndarray, domain: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Fill domain cells lacking data with their latitude band's mean, per month."""
    out = field.copy()
    bands = np.floor((lats + 90) / FILL_BAND_DEG).astype(int)
    for month in out:
        for band in np.unique(bands):
            rows = bands == band
            block = month[rows]
            have = np.isfinite(block)
            gaps = domain[rows] & ~have
            if gaps.any() and have.any():
                block[gaps] = block[have].mean()
                month[rows] = block
    return out


def smooth_spatial(
    field: np.ndarray, domain: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """~SMOOTHING_KM Gaussian over the domain (normalized, so coasts aren't
    pulled toward zero), wrapping in longitude, widening in degrees of
    longitude toward the poles to stay roughly isotropic in distance."""
    from scipy.ndimage import gaussian_filter1d

    sigma_lat = SMOOTHING_KM / (KM_PER_DEG * COMBINE_RES)
    cos_lat = np.maximum(np.cos(np.radians(lats)), 0.05)

    def smooth(a: np.ndarray) -> np.ndarray:
        a = gaussian_filter1d(a, sigma_lat, axis=0, mode="nearest")
        return np.stack(
            [
                gaussian_filter1d(row, min(sigma_lat / c, a.shape[1] / 4), mode="wrap")
                for row, c in zip(a, cos_lat, strict=True)
            ]
        )

    out = np.empty_like(field)
    for i, month in enumerate(field):
        have = domain & np.isfinite(month)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = smooth(np.where(have, month, 0.0)) / smooth(have.astype(float))
        out[i] = np.where(domain, smoothed, np.nan)
    return out


def infill_from_most_complete(deltas: list[np.ndarray], lats: np.ndarray) -> np.ndarray:
    """Stack the datasets, filling each one's gaps from the most complete one.

    Shifting the fill by the gappy dataset's mean difference from the complete
    one (per month, over cells both cover) keeps each dataset's overall level,
    so the average neither jumps where a dataset's coverage ends nor leans
    toward whichever dataset happens to cover the poorly observed regions.
    """
    stack = np.stack(deltas)
    coverage = np.isfinite(stack).sum(axis=(1, 2, 3))
    complete = stack[int(np.argmax(coverage))]
    weights = np.broadcast_to(np.cos(np.radians(lats))[:, None], complete.shape[1:])
    for member in stack:
        for month, full in zip(member, complete, strict=True):
            both = np.isfinite(month) & np.isfinite(full)
            if both.any() and not both.all():
                bias = np.average((month - full)[both], weights=weights[both])
                gaps = ~np.isfinite(month)
                month[gaps] = full[gaps] + bias
    return stack


# Reconstructions reach pre-industrial from 1991–2020, the period every source
# covers fully, and each source reaches 1991–2020 by its own Δ. That keeps the
# chain off OISST's "1971–2000" climatology, which predates OISST and is only
# partly built from it (its 1991–2020 anomaly mean is +0.09 °C globally, vs
# ERA5's +0.24) — and it's how Copernicus links ERA5 to 1850–1900.
PREINDUSTRIAL_ANCHOR = "1991-2020"
# Reconstruction field → the source Δ fields it's added to.
PREINDUSTRIAL_FIELDS = {"sst": ("sst", "oisst"), "t2m": ("t2m",)}


def build_preindustrial() -> None:
    period = baselines.BASELINES["preindustrial"].years
    anchor = baselines.BASELINES[PREINDUSTRIAL_ANCHOR].years
    lats = np.arange(-90 + COMBINE_RES / 2, 90, COMBINE_RES)
    lons = np.arange(COMBINE_RES / 2, 360, COMBINE_RES)

    for field, targets in PREINDUSTRIAL_FIELDS.items():
        print(f"── {field}: {PREINDUSTRIAL_ANCHOR} minus {period[0]}–{period[1]}")
        deltas, domain = [], np.zeros((lats.size, lons.size), dtype=bool)
        for rec in (r for r in RECONSTRUCTIONS if r.field == field):
            da, years, months = open_monthly(download(rec.url), rec.var)
            ref = period_monthly_mean(da, years, months, anchor)
            delta = ref - period_monthly_mean(da, years, months, period)
            native_lats = da["latitude"].values
            covered = np.isfinite(delta).all(axis=0).mean()
            role = "" if rec.averaged else "  (cross-check only)"
            print(
                f"  {rec.name:15} global {global_mean(delta, native_lats):+.3f} °C, "
                f"{covered:.0%} of cells covered{role}"
            )
            if not rec.averaged:
                continue
            grid = (native_lats, da["longitude"].values, lats, lons)
            deltas.append(to_grid(delta, *grid))
            domain |= np.isfinite(to_grid(ref, *grid)).any(axis=0)

        stack = infill_from_most_complete(deltas, lats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # cells no dataset has
            combined = np.nanmean(stack, axis=0)
        if field == "t2m":
            domain[:] = True
        gaps = domain & ~np.isfinite(combined).all(axis=0)
        print(f"  filled {gaps.sum() / domain.sum():.1%} of cells from latitude bands")
        combined = smooth_spatial(fill_from_bands(combined, domain, lats), domain, lats)
        print(f"  {'combined':15} global {global_mean(combined, lats):+.3f} °C")
        combined = to_grid(combined, lats, lons, _OISST_LATS, _OISST_LONS)
        for target in targets:
            to_anchor = baselines.load_delta(PREINDUSTRIAL_ANCHOR, target)
            write_delta("preindustrial", target, to_anchor + fill_nearest(combined))


# ---------------------------------------------------------------- offsets


def ensure_oisst_mask() -> None:
    path = baselines.FIELD_DIR / "valid-oisst-sst.npy"
    if path.exists():
        return
    import h5py

    with h5py.File(download(OISST_DAY_URL), "r") as hdf:
        land = hdf["sst"][0, 0] == -999
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ~land)
    print(f"✓ {path}")


def write_offsets() -> None:
    """Average every available Δ field over each source's regions."""
    ensure_oisst_mask()
    lon_2d, lat_2d = np.meshgrid(_OISST_LONS, _OISST_LATS)
    table: baselines.OffsetTable = {}
    for baseline_id in baselines.BASELINES:
        for (source, dataset), variable in baselines.ANOMALY_VARIABLE.items():
            field = baselines.delta_field(baseline_id, source, dataset)
            if not baselines.delta_path(baseline_id, field).exists():
                continue
            delta = baselines.load_delta(baseline_id, field)
            valid = baselines.load_valid_mask(source, variable)
            table.setdefault(baseline_id, {}).setdefault(source, {})[dataset] = (
                baselines.regional_offsets(delta, valid, lat_2d, lon_2d)
            )
    baselines.OFFSETS_PATH.write_text(json.dumps(table, indent=1) + "\n")
    print(f"✓ {baselines.OFFSETS_PATH}")
    for baseline_id, by_source in table.items():
        for source, by_dataset in by_source.items():
            for dataset, by_region in by_dataset.items():
                annual = by_region["global"][0]
                print(f"  {baseline_id:>14} {source}.{dataset}: global {annual:+.3f}")


# The globe's copy of each Δ field: its five seasonal coefficients at full
# resolution, stacked as tiles in one 1440×3600 image. A 0.002 °C step keeps
# any day's Δ (five rounded terms) within 0.005 °C. Coarser grids save little
# and miss sharp features at coasts and sea-ice edges by >0.1 °C.
DELTA_ENCODING = Rg16(scale=0.002, offset=-32.768)


def write_atlases(publish_dir: Path) -> None:
    """Publish each Δ field's seasonal coefficients as an RG16 atlas + manifest.

    Written to ``{publish_dir}/baselines/`` (upload-to-s3.py publishes it as
    ``sea-surface-temp/baselines/``). Tile k from the top holds coefficient k
    of [a0, a1, b1, a2, b2] (see baselines.py); each tile is north up with
    longitude 0→360° left to right, like the data textures. The manifest maps
    every (baseline, source, anomaly dataset) to its atlas, so the site needn't
    know which sources share a field.
    """
    out_dir = publish_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, dict[str, dict[str, str]]] = {}
    for baseline_id in baselines.BASELINES:
        for source, dataset in baselines.ANOMALY_VARIABLE:
            field = baselines.delta_field(baseline_id, source, dataset)
            if not baselines.delta_path(baseline_id, field).exists():
                continue
            name = f"delta-{field}-{baseline_id}.webp"
            fields.setdefault(baseline_id, {}).setdefault(source, {})[dataset] = name
            if (out_dir / name).exists():
                continue
            coefficients = baselines.seasonal_coefficients(
                baselines.load_delta(baseline_id, field)
            )
            # encode() flips rows (south→north in, north up out), so stack the
            # tiles bottom-first to land tile 0 on top.
            atlas = np.concatenate(list(coefficients[::-1]), axis=0)
            save_lossless_webp(DELTA_ENCODING.encode(atlas), out_dir / name)
            print(f"✓ {out_dir / name}")
    manifest = {
        "encoding": DELTA_ENCODING.as_metadata(),
        "terms": ["a0", "a1", "b1", "a2", "b2"],
        "tile_width": len(_OISST_LONS),
        "tile_height": len(_OISST_LATS),
        "fields": fields,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"✓ {out_dir / 'manifest.json'}")


BUILDERS = {"1991-2020": build_1991_2020, "preindustrial": build_preindustrial}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", choices=sorted(BUILDERS), action="append")
    parser.add_argument(
        "--offsets-only",
        action="store_true",
        help="Skip building Δ fields; just recompute offsets.json and atlases",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("./maps"),
        help="Where to write the globe's Δ atlases (under baselines/)",
    )
    args = parser.parse_args(argv)

    if not args.offsets_only:
        for baseline_id in args.baseline or sorted(BUILDERS):
            BUILDERS[baseline_id]()
        shutil.rmtree(args.publish_dir / "baselines", ignore_errors=True)
    write_offsets()
    write_atlases(args.publish_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
