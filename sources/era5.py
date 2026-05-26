# SPDX-License-Identifier: MIT
"""ECMWF ERA5 reanalysis source (daily-mean SST + 2m air temp).

Pulls the pre-computed daily mean (UTC day) for both variables from the
Copernicus CDS ``derived-era5-single-levels-daily-statistics`` dataset,
resamples the native 721×1440 ERA5 grid onto the 720×1440 OISST grid at
fetch time, and stores the result locally as ``era5-YYYYMMDD.nc`` in
Kelvin — unit conversion happens in :meth:`get_data_array` so the archive
matches ERA5's native convention.

Why daily-mean and not a fixed UTC timestamp? Earlier versions sampled at
12:00 UTC, which made the globe visibly hotter on the sun-facing side and
introduced a small but real bias that varied with longitude. The daily
mean (24-hour average over UTC) matches OISST's daily-mean L4 product so
the two sources are directly comparable.

Why resample at fetch time and not aggregate time? The same regions/masks
work unchanged for OISST and ERA5 once they share a grid, and the resampled
file is ~30% smaller than the native one (no extra row at each pole).
"""

from __future__ import annotations

import asyncio
import calendar
import datetime
import functools
import re
import tempfile
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from .base import DataSource, DatasetSpec

if TYPE_CHECKING:
    import aiohttp
    import xarray as xr  # noqa: F401


# OISST cell-center grid (canonical target for both sources).
# lats: -89.875 .. 89.875 step 0.25 (720 points, ascending)
# lons:   0.125 .. 359.875 step 0.25 (1440 points)
_OISST_LATS = np.arange(-89.875, 90.0, 0.25)
_OISST_LONS = np.arange(0.125, 360.0, 0.25)


# Reuses the OISST SST cmap so the two sources look interchangeable.
_ERA5_SST_CMAP: list[list[Any]] = [
    [0, "darkblue"],
    [20, "white"],
    [24, "yellow"],
    [28, "orange"],
    [30, "red"],
    [32, "darkred"],
    [35, "#470000"],
]

# 2m air-temperature shares SST's 0–32°C anchors so switching SST↔t2m gives
# directly comparable colors in the populated temperature range; the upper end
# extends to 45°C for hot land that SST never sees. Sub-zero regions
# (Antarctica, NH winter, polar night) clip to darkblue — those values are all
# uniformly "very cold" and dedicating cmap range to resolving −10°C vs −50°C
# would squash detail in the 0–35°C band where most populated land lives.
_ERA5_T2M_CMAP: list[list[Any]] = [
    [0,  "darkblue"],
    [20, "white"],
    [24, "yellow"],
    [28, "orange"],
    [30, "red"],
    [32, "darkred"],
    [45, "#470000"],
]

# SST anomaly cmap, matching OISST's convention so the two sources' anom
# overlays look identical at a glance. Asymmetric (warm side extends to +7°C,
# cold side stops at -3°C) — anomalies skew warm under the current climate.
_ERA5_SST_ANOM_CMAP: list[list[Any]] = [
    [-3,   "darkblue"],
    [-0.5, "lightblue"],
    [0,    "white"],
    [1.5,  "yellow"],
    [3,    "red"],
    [5,    "darkred"],
    [7,    "#470000"],
]

# 2 m air-temperature anomaly cmap. Wider range than SST because land has
# much larger day-to-day variability: heat waves push +10°C, cold snaps
# −15°C; extreme weather events can hit ±20°C. Same diverging structure as
# SST anom so the colors still read intuitively.
_ERA5_T2M_ANOM_CMAP: list[list[Any]] = [
    [-15, "darkblue"],
    [-4,  "lightblue"],
    [0,   "white"],
    [4,   "yellow"],
    [10,  "red"],
    [15,  "darkred"],
    [20,  "#470000"],
]


# Filenames for the daily climatologies built by
# scripts/build_era5_climatology.py. One per variable, both live under
# ./era5-archive/climatology/ locally; CI fetches the same files from S3
# before the ERA5 step.
CLIMATOLOGY_FILENAMES: dict[str, str] = {
    "sst": "era5-climatology-1971-2000.nc",
    "t2m": "era5-t2m-climatology-1971-2000.nc",
}


@functools.lru_cache(maxsize=2)
def _load_climatology(path_str: str, var_name: str) -> np.ndarray:
    """Open a climatology NetCDF and return its data as a NumPy array.

    Cached per-process — aggregate_archive.py spawns many worker processes,
    each loads each climatology once and reuses it across every aggregated
    NetCDF. ~140 MB resident per (worker × climatology).

    Returns shape (366, 720, 1440) float32 in °C. SST climatology has NaN
    over land; t2m is defined everywhere.
    """
    import xarray as xr

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"ERA5 {var_name} climatology not found at {p}. Run "
            f"scripts/build_era5_climatology.py --variable {var_name} to "
            f"build it, or download from "
            f"s3://climate-change-assets/sea-surface-temp/climatology/{p.name}."
        )
    with xr.open_dataset(p) as ds:
        return np.asarray(ds[f"{var_name}_climatology"].values, dtype=np.float32)


def _leap_year_doy(date: datetime.date) -> int:
    """Day-of-year in a leap-year calendar (1..366) — matches the climatology axis.

    Same convention as scripts/build_era5_climatology.py: Jan-Feb keep their
    natural DOY; Mar 1 always maps to DOY 61. Feb 29 → 60 (leap years only).
    """
    natural = date.timetuple().tm_yday
    if calendar.isleap(date.year) or date.month < 3:
        return natural
    return natural + 1


class Era5FetchError(Exception):
    """Raised when a CDS request fails after retries."""


_FILENAME_RE = re.compile(r"era5-(\d{4})(\d{2})(\d{2})\.nc$")


def _resample_to_oisst_grid(ds: "xr.Dataset") -> "xr.Dataset":
    """Linearly interpolate ERA5 (90→-90 lat, 0→359.75 lon) onto OISST cell centers.

    Pads one wrap-around column at lon=360 so OISST's max cell center (359.875)
    falls inside the source longitude range. The output dims are
    ``(valid_time, latitude, longitude) = (1, 720, 1440)``.
    """
    import xarray as xr

    # Pad longitude to enable interpolation up to 359.875 (just under 360).
    # Drop the time-invariant scalar coords first; assign_coords/concat trip on
    # them otherwise.
    extra = ds.isel(longitude=0).assign_coords(longitude=360.0)
    padded = xr.concat([ds, extra], dim="longitude")

    return padded.interp(
        latitude=_OISST_LATS,
        longitude=_OISST_LONS,
        method="linear",
    )


def _do_cds_retrieve(date: datetime.date, out_path: Path) -> None:
    """Synchronous CDS fetch of one UTC daily-mean + resample.

    The ``derived-era5-single-levels-daily-statistics`` endpoint returns a
    zip bundle (one NetCDF per requested variable) regardless of
    ``download_format`` — it logs ``"Download format not supported for this
    dataset. Defaulting to as_source."`` and ignores the request. Handle
    the zip explicitly: extract → ``xr.merge`` → resample → write.
    """
    import zipfile

    import cdsapi
    import xarray as xr

    request = {
        "product_type": "reanalysis",
        "variable": ["sea_surface_temperature", "2m_temperature"],
        "year": f"{date.year:04}",
        "month": f"{date.month:02}",
        "day": f"{date.day:02}",
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
    }

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        zip_path = tmpdir / "response.zip"

        client = cdsapi.Client()
        try:
            client.retrieve(
                "derived-era5-single-levels-daily-statistics",
                request,
                str(zip_path),
            )
        except Exception as e:
            raise Era5FetchError(f"CDS retrieve failed for {date}: {e}") from e

        # Extract the per-variable NetCDFs from the zip and merge into one
        # dataset whose shape matches the previous single-levels API.
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        nc_files = sorted(tmpdir.glob("*.nc"))
        if not nc_files:
            raise Era5FetchError(
                f"CDS response for {date} contained no NetCDFs: {[n.name for n in tmpdir.iterdir()]}"
            )
        # Load each file fully into memory then close — keeping the dataset
        # objects "live" past tmpdir cleanup with `load()` avoids lazy-read
        # IOErrors after the underlying files are deleted.
        parts = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                parts.append(ds.load())
        merged = xr.merge(parts, compat="override")
        resampled = _resample_to_oisst_grid(merged).load()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Compress data variables with zlib so the local archive stays small
        # (~2 MB/day vs ~16 MB uncompressed). Coords stay uncompressed — they're
        # tiny and netcdf libraries trip on chunking 1D coord arrays.
        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in resampled.data_vars
        }
        resampled.to_netcdf(out_path, encoding=encoding)


class Era5Source(DataSource):
    id = "era5"
    grid_shape = (720, 1440)  # post-resample, matches OISST masks

    datasets = {
        "sst": DatasetSpec(
            id="sst",
            cmap_def=_ERA5_SST_CMAP,
            title_template="{date}\nERA5 Sea Surface Temp, °C",
            equirect_basename="era5-sst",
            map_basename="era5-sst",
        ),
        "t2m": DatasetSpec(
            id="t2m",
            cmap_def=_ERA5_T2M_CMAP,
            title_template="{date}\nERA5 2 m Air Temp, °C",
            equirect_basename="era5-t2m",
            map_basename="era5-t2m",
        ),
        "sst_anom": DatasetSpec(
            id="sst_anom",
            cmap_def=_ERA5_SST_ANOM_CMAP,
            title_template="{date}\nERA5 SST Anomaly vs 1971–2000 Mean, °C",
            equirect_basename="era5-sst-anom",
            map_basename="era5-sst-anom",
        ),
        "t2m_anom": DatasetSpec(
            id="t2m_anom",
            cmap_def=_ERA5_T2M_ANOM_CMAP,
            title_template="{date}\nERA5 2 m Air Temp Anomaly vs 1971–2000 Mean, °C",
            equirect_basename="era5-t2m-anom",
            map_basename="era5-t2m-anom",
        ),
    }

    archive_root = Path("./era5-archive")
    climatology_dir = archive_root / "climatology"

    @classmethod
    def climatology_path_for(cls, variable: str) -> Path:
        return cls.climatology_dir / CLIMATOLOGY_FILENAMES[variable]

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
            / f"era5-{date.year:04}{date.month:02}{date.day:02}.nc"
        )

    async def fetch(
        self,
        date: datetime.date,
        session: aiohttp.ClientSession,  # unused; kept for interface parity
        semaphore: Any,  # unused; CDS imposes its own rate limit
    ) -> "xr.Dataset":
        """Download + resample one date. Returns an in-memory xarray Dataset.

        Caches under ``./era5-archive/`` so re-runs (e.g. the daily cron's
        last-N-days loop) don't re-hit the CDS queue.
        """
        out_path = self.archive_path(self.archive_root, date)
        if not out_path.exists():
            print(f"Requesting ERA5 data for {date} (queues at CDS, ~25s)...")
            await asyncio.to_thread(_do_cds_retrieve, date, out_path)
            print(f"✅Got ERA5 {date}")
        else:
            print(f"Using cached ERA5 archive {out_path}")
        # Open the cached file as a fresh dataset; caller is responsible for
        # closing via the get_data_array / open_local lifecycle. We load() so
        # the file handle can be released immediately.
        import xarray as xr

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
        lat_1d = raw["latitude"].values
        lon_1d = raw["longitude"].values
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        return lat_2d, lon_2d

    # Map derived anomaly datasets to (raw NetCDF var, climatology variable).
    # sst_anom = raw sst minus sst climatology; t2m_anom = raw t2m minus
    # t2m climatology.
    _ANOM_PARENT: dict[str, str] = {"sst_anom": "sst", "t2m_anom": "t2m"}

    def get_data_array(
        self,
        raw: "xr.Dataset",
        dataset_id: str,
        **_kwargs: Any,
    ) -> np.ma.MaskedArray:
        """Return a 2D masked array in °C on the OISST grid.

        ERA5 SST is NaN over land; ``np.ma.masked_invalid`` converts that to a
        proper mask matching OISST's masked-array convention so
        :func:`regions.aggregate` can treat the two sources interchangeably.

        For ``sst_anom`` / ``t2m_anom``: returns (raw − climatology[doy])
        using the same DOY-indexing convention as the climatology builder
        (leap-year calendar: Mar 1 always at DOY 61; Feb 29 at DOY 60 from
        leap years only).
        """
        parent = self._ANOM_PARENT.get(dataset_id)
        var_name = parent if parent is not None else dataset_id
        var = raw[var_name]
        # Drop singleton time/number dims, keep (lat, lon).
        arr_k = np.asarray(var.squeeze().values, dtype=np.float32)
        arr_c = arr_k - 273.15

        if parent is not None:
            tdim = "valid_time" if "valid_time" in raw.dims else "time"
            ts = np.datetime64(raw[tdim].values.flat[0], "D").astype("O")
            doy = _leap_year_doy(datetime.date(ts.year, ts.month, ts.day))
            clim = _load_climatology(str(self.climatology_path_for(parent)), parent)
            arr_c = arr_c - clim[doy - 1]

        return np.ma.masked_invalid(arr_c)
