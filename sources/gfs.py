# SPDX-License-Identifier: MIT
"""NOAA GFS operational 2 m air-temperature source (daily mean / max / min).

GFS is an operational *forecast* model, not a reanalysis: NCEP runs it 4×/day
and the 0.25° GRIB2 output lands on AWS within ~1–2 h, so it fills ERA5's
6–7-day reporting gap on the globe. The trade-off is that GFS is upgraded
periodically (resolution/physics), so its multi-year record has
discontinuities — it is the "today / recent" layer, **not** a source for the
multi-decade trend graphs, where ERA5 stays authoritative.

Daily aggregation. GFS ``TMP:2 m above ground`` is *instantaneous*. We take
eight 3-hourly slices (f000…f021 of the 00Z cycle, covering the 24 h UTC day)
and reduce per cell to mean / max / min:

* ``t2m-mean`` — directly comparable to the ERA5/OISST daily-mean globe.
* ``t2m-max`` / ``t2m-min`` — the diurnal extremes (heat stress, A/C demand)
  a daily mean hides.

These three are one *variable* (``t2m``) along a *statistic* axis, so the
frontend renders them as a min/mean/max button group within the GFS source
rather than three unrelated datasets (see ``DatasetSpec.variable/statistic``).

Anomalies use the **ERA5 1971–2000 climatology** (mean/max/min) as the
baseline, not a GFS self-climatology: GFS only exists from 2015 and isn't
stationary, so a GFS baseline would be both short and non-comparable to ERA5
anomalies. The small GFS-vs-ERA5 mean-state bias is acceptable for visual
comparability (the analogue of Climate Reanalyzer using a CFSR baseline for its
GFS maps). The max/min climatologies are built by
``scripts/build_era5_climatology.py`` from the CDS daily-statistics API; the
mean reuses the existing ERA5 daily-mean t2m climatology.

Data access is a single path: anonymous reads from the ``noaa-gfs-bdp-pds`` AWS
Open Data bucket, byte-range-subset to just the 2 m TMP record via each file's
``.idx`` sidecar (~1–2 MB/slice instead of ~500 MB). NOMADS is avoided — it
keeps only ~10 days and rate-limits hard.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import re
import tempfile
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from .base import DataSource, DatasetSpec
from .era5 import (
    Era5Source,
    _leap_year_doy,
    _load_climatology,
    _resample_to_oisst_grid,
)
from .oisst import DataFetchError

if TYPE_CHECKING:
    import aiohttp
    import xarray as xr  # noqa: F401


# ---------------------------------------------------------------------------
# AWS GRIB access
# ---------------------------------------------------------------------------

_S3_BUCKET = "noaa-gfs-bdp-pds"
# 00Z cycle forecast hours spanning the 24 h UTC day (valid 00,03,…,21Z). One
# model run keeps the eight slices internally consistent — the same choice
# Climate Reanalyzer makes for its "1-day Avg" maps.
_FORECAST_HOURS: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21)
# Substring identifying the 2 m temperature record in a GFS .idx line, e.g.
# "57:38500000:d=2026062100:TMP:2 m above ground:3 hour fcst:".
_TMP_2M_MATCH = ":TMP:2 m above ground:"


class GfsFetchError(DataFetchError):
    """Raised when GFS data can't be retrieved for a date.

    Subclasses ``DataFetchError`` so ``pipeline.get_temp_for_date`` treats a
    missing GFS day like a missing OISST day (NaN + continue) in graph mode.
    """


def _s3_client() -> Any:
    """Anonymous S3 client for the public GFS Open Data bucket."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _grib_subset(s3: Any, key: str) -> bytes:
    """Return just the 2 m TMP GRIB record from ``key`` using its .idx sidecar.

    Reads the ``.idx`` text index, finds the 2 m-temperature record, and issues
    a ranged GET for exactly that record's bytes.
    """
    from botocore.exceptions import ClientError

    try:
        idx_body = s3.get_object(Bucket=_S3_BUCKET, Key=key + ".idx")["Body"].read()
    except ClientError as e:
        raise GfsFetchError(f"GFS index missing: s3://{_S3_BUCKET}/{key}.idx ({e})") from e

    lines = idx_body.decode("utf-8").splitlines()
    for i, line in enumerate(lines):
        if _TMP_2M_MATCH in line:
            start = int(line.split(":")[1])
            if i + 1 < len(lines):
                rng = f"bytes={start}-{int(lines[i + 1].split(':')[1]) - 1}"
            else:
                rng = f"bytes={start}-"
            obj = s3.get_object(Bucket=_S3_BUCKET, Key=key, Range=rng)
            return obj["Body"].read()
    raise GfsFetchError(f"No 2 m TMP record in s3://{_S3_BUCKET}/{key}.idx")


def _read_slice(blob: bytes, tmpdir: Path, fhr: int) -> "xr.Dataset":
    """Open a single-record GRIB2 blob with cfgrib and return its dataset."""
    import xarray as xr

    grib_path = tmpdir / f"gfs-f{fhr:03d}.grib2"
    grib_path.write_bytes(blob)
    # indexpath="" keeps cfgrib from writing a sibling .idx into the (possibly
    # read-only) data dir; the temp dir is discarded anyway.
    return xr.open_dataset(
        grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
    )


def _build_daily_nc(date: datetime.date, out_path: Path) -> None:
    """Fetch the day's 8 slices, reduce to mean/max/min, resample, and cache.

    Writes ``gfs-YYYYMMDD.nc`` with three Kelvin variables (``t2m_mean``,
    ``t2m_max``, ``t2m_min``) on the OISST 720×1440 grid.
    """
    import xarray as xr

    s3 = _s3_client()
    run = f"{date.year:04}{date.month:02}{date.day:02}"

    slices: list[np.ndarray] = []
    lat = lon = None
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for fhr in _FORECAST_HOURS:
            key = f"gfs.{run}/00/atmos/gfs.t00z.pgrb2.0p25.f{fhr:03d}"
            ds = _read_slice(_grib_subset(s3, key), tmpdir, fhr)
            da = ds["t2m"] if "t2m" in ds else ds[list(ds.data_vars)[0]]
            slices.append(np.asarray(da.values, dtype=np.float32))
            if lat is None:
                lat, lon = da["latitude"].values, da["longitude"].values

    stack = np.stack(slices)  # (8, 721, 1440), Kelvin
    daily = xr.Dataset(
        {
            "t2m_mean": (("latitude", "longitude"), stack.mean(axis=0)),
            "t2m_max": (("latitude", "longitude"), stack.max(axis=0)),
            "t2m_min": (("latitude", "longitude"), stack.min(axis=0)),
        },
        coords={"latitude": lat, "longitude": lon},
    ).assign_coords(valid_time=np.datetime64(date.isoformat()))

    resampled = _resample_to_oisst_grid(daily).load()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for v in resampled.data_vars
    }
    # Write to a temp sibling then atomically rename, so a crash mid-write can't
    # leave a corrupt file at out_path (which the fetch phase's .exists() check
    # would otherwise treat as a valid checkpoint and never rebuild).
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    resampled.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# Colormaps — kept identical to ERA5's t2m maps so GFS↔ERA5 mean comparisons,
# and the three statistics among themselves, all share one scale. (max values
# above 45 °C clip to the darkest red; dedicating range above that would squash
# the populated 0–35 °C band.)
# ---------------------------------------------------------------------------

_GFS_T2M_CMAP: list[list[Any]] = [
    [0, "darkblue"],
    [20, "white"],
    [24, "yellow"],
    [28, "orange"],
    [32, "#ed1c24"],  # red
    [36, "#c41e3a"],  # crimson
    [40, "#9e1b4a"],  # dark raspberry (red-leaning purple)
    [44, "#7c1d1d"],  # brick
    [48, "#4a0d0d"],  # dark maroon
    [52, "#120000"],  # near-black
]

_GFS_T2M_ANOM_CMAP: list[list[Any]] = [
    [-10, "darkblue"],
    [-6, "#2166ac"],
    [-3, "#4393c3"],
    [-1.5, "lightblue"],
    [-0.5, "#e2eff9"],
    [0, "white"],
    [0.5, "#fffac0"],
    [1.0, "#fff176"],
    [2.0, "#ffe000"],
    [3.0, "#ffc107"],
    [4.0, "#ff9800"],
    [6.0, "#ff5722"],
    [9.0, "#e53935"],
    [13, "darkred"],
    [16, "#470000"],
]


# Dataset ids use underscores (t2m_mean) to match ERA5's convention and the
# frontend DatasetId — critically, the cache key + export_timeseries.py only
# allow [a-z0-9_] in the dataset token, so a hyphenated id would drop every GFS
# time-series entry. Texture *filenames* still hyphenate via equirect_basename
# (gfs-t2m-mean), which the S3 index normalizes back to underscores.
def _temp_spec(stat: str, label: str) -> DatasetSpec:
    return DatasetSpec(
        id=f"t2m_{stat}",
        cmap_def=_GFS_T2M_CMAP,
        title_template=f"{{date}}\nGFS Daily {label} 2 m Air Temp, °C",
        equirect_basename=f"gfs-t2m-{stat}",
        map_basename=f"gfs-t2m-{stat}",
        variable="t2m",
        statistic=stat,
        kind="temp",
    )


def _anom_spec(stat: str, label: str) -> DatasetSpec:
    return DatasetSpec(
        id=f"t2m_{stat}_anom",
        cmap_def=_GFS_T2M_ANOM_CMAP,
        title_template=(
            f"{{date}}\nGFS Daily {label} 2 m Air Temp Anomaly vs ERA5 1971–2000, °C"
        ),
        equirect_basename=f"gfs-t2m-{stat}-anom",
        map_basename=f"gfs-t2m-{stat}-anom",
        variable="t2m",
        statistic=stat,
        kind="anomaly",
    )


_FILENAME_RE = re.compile(r"gfs-(\d{4})(\d{2})(\d{2})\.nc$")


class GfsSource(DataSource):
    id = "gfs"
    grid_shape = (720, 1440)  # post-resample, matches OISST masks
    archive_root = Path("./gfs-archive")

    datasets = {
        "t2m_mean": _temp_spec("mean", "Mean"),
        "t2m_max": _temp_spec("max", "Max"),
        "t2m_min": _temp_spec("min", "Min"),
        "t2m_mean_anom": _anom_spec("mean", "Mean"),
        "t2m_max_anom": _anom_spec("max", "Max"),
        "t2m_min_anom": _anom_spec("min", "Min"),
    }

    # Anomaly dataset → (raw NetCDF variable, ERA5 climatology variable). The
    # mean reuses the existing era5-t2m climatology ("t2m"); max/min need their
    # own climatology files built by scripts/build_era5_climatology.py.
    _ANOM_INFO: dict[str, tuple[str, str]] = {
        "t2m_mean_anom": ("t2m_mean", "t2m"),
        "t2m_max_anom": ("t2m_max", "t2m_max"),
        "t2m_min_anom": ("t2m_min", "t2m_min"),
    }

    @staticmethod
    def date_from_filename(path: Path) -> tuple[int, int, int] | None:
        m = _FILENAME_RE.match(path.name)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    @staticmethod
    def archive_path(archive_root: Path, date: datetime.date) -> Path:
        return (
            archive_root
            / f"{date.year:04}"
            / f"gfs-{date.year:04}{date.month:02}{date.day:02}.nc"
        )

    async def fetch(
        self,
        date: datetime.date,
        session: aiohttp.ClientSession,  # unused; AWS access is via boto3
        semaphore: Any,
    ) -> "xr.Dataset":
        """Download + aggregate one date. Returns an in-memory xarray Dataset.

        Caches ``gfs-YYYYMMDD.nc`` so re-runs (the daily cron's last-N-days
        loop, region back-aggregation) don't re-download the GRIB slices.
        """
        import xarray as xr

        out_path = self.archive_path(self.archive_root, date)
        if not out_path.exists():
            async with semaphore:
                print(f"Building GFS daily mean/max/min for {date} from AWS...")
                await asyncio.to_thread(_build_daily_nc, date, out_path)
                print(f"✅Got GFS {date}")
        else:
            print(f"Using cached GFS archive {out_path}")

        with xr.open_dataset(out_path) as ds:
            return ds.load()

    def open_local(self, path: Path) -> AbstractContextManager["xr.Dataset"]:
        @contextmanager
        def _opener() -> Iterator["xr.Dataset"]:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                yield ds.load()

        return _opener()

    def latlon_2d(self, raw: "xr.Dataset") -> tuple[np.ndarray, np.ndarray]:
        lon_2d, lat_2d = np.meshgrid(raw["longitude"].values, raw["latitude"].values)
        return lat_2d, lon_2d

    def get_data_array(
        self,
        raw: "xr.Dataset",
        dataset_id: str,
        **_kwargs: Any,
    ) -> np.ma.MaskedArray:
        """Return a 2D masked array in °C on the OISST grid.

        Temperature datasets return the statistic converted K→°C. Anomaly
        datasets subtract the ERA5 climatology for the day-of-year, using the
        same leap-year DOY convention as the climatology builder.
        """
        anom = self._ANOM_INFO.get(dataset_id)
        var_name = anom[0] if anom is not None else dataset_id.replace("-", "_")
        arr_c = np.asarray(raw[var_name].squeeze().values, dtype=np.float32) - 273.15

        if anom is not None:
            _, clim_var = anom
            ts = np.datetime64(raw["valid_time"].values, "D").astype("O")
            doy = _leap_year_doy(datetime.date(ts.year, ts.month, ts.day))
            clim_path = Era5Source.climatology_path_for(clim_var)
            arr_c = arr_c - _load_climatology(str(clim_path), clim_var)[doy - 1]

        return np.ma.masked_invalid(arr_c)
