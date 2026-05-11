# SPDX-License-Identifier: MIT
"""ECMWF ERA5 reanalysis source (single-levels SST + 2m air temp).

Pulls one timestamp per day (12:00 UTC) from the Copernicus CDS, resamples
the native 721×1440 ERA5 grid onto the 720×1440 OISST grid at fetch time,
and stores the result locally as ``era5-YYYYMMDD.nc`` (still in Kelvin —
unit conversion happens in :meth:`get_data_array` so the archive matches
ERA5's native convention).

Why resample at fetch time and not aggregate time? The same regions/masks
work unchanged for OISST and ERA5 once they share a grid, and the resampled
file is ~30% smaller than the native one (no extra row at each pole).

Why fixed 12:00 UTC instead of a daily mean? One CDS request per day, one
license. A proper daily mean would need either 24× the data (averaging
locally) or the separate ``derived-era5-single-levels-daily-statistics``
dataset (extra license). The ~0.1°C systematic bias from sampling at noon
UTC is small enough for the Trends-tab use case; revisit if the smoke
comparison against OISST exceeds 0.5°C globally.
"""

from __future__ import annotations

import asyncio
import datetime
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
    """Synchronous CDS fetch of one timestamp + resample. Caller wraps in a thread."""
    import cdsapi
    import xarray as xr

    request = {
        "product_type": "reanalysis",
        "variable": ["sea_surface_temperature", "2m_temperature"],
        "year": f"{date.year:04}",
        "month": f"{date.month:02}",
        "day": f"{date.day:02}",
        "time": "12:00",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        client = cdsapi.Client()
        try:
            client.retrieve("reanalysis-era5-single-levels", request, str(tmp_path))
        except Exception as e:
            raise Era5FetchError(f"CDS retrieve failed for {date}: {e}") from e

        with xr.open_dataset(tmp_path) as native:
            # Materialize (otherwise lazy reads break after the file is deleted)
            # and resample in one pass.
            resampled = _resample_to_oisst_grid(native).load()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Compress data variables with zlib so the local archive stays small
        # (~2 MB/day vs ~16 MB uncompressed). Coords stay uncompressed — they're
        # tiny and netcdf libraries trip on chunking 1D coord arrays.
        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for v in resampled.data_vars
        }
        resampled.to_netcdf(out_path, encoding=encoding)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    }

    archive_root = Path("./era5-archive")

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
        """
        var = raw[dataset_id]
        # Drop singleton time/number dims, keep (lat, lon).
        arr_k = np.asarray(var.squeeze().values, dtype=np.float32)
        arr_c = arr_k - 273.15
        return np.ma.masked_invalid(arr_c)
