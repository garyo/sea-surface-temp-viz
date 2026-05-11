# SPDX-License-Identifier: MIT
"""NOAA OISST v2.1 daily 0.25° gridded SST + anomaly source.

Wraps the HTTP fetch from ``ncei.noaa.gov`` and the HDF5 extraction that the
pipeline used to do inline. The raw handle is an open ``h5py.File``.
"""

from __future__ import annotations

import datetime
import io
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import h5py
import numpy as np

from .base import DataSource, DatasetSpec

if TYPE_CHECKING:
    import aiohttp


class DataFetchError(Exception):
    """Raised when all OISST URL candidates fail."""


_OISST_URL_BASE = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr"
)


def _rel_area(lat: np.ndarray, _lon: np.ndarray) -> np.ndarray:
    """cos(latitude) — the relative area of a 0.25° cell vs. one at the equator."""
    return np.cos(np.radians(lat))


async def _try_fetch(urls: list[str], session: aiohttp.ClientSession) -> bytes:
    """Return body of the first URL that returns 200; raise if all fail."""
    for url in urls:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                print(f"Failed to fetch {url}: status={response.status}")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    raise DataFetchError("All URL fetches failed")


# Colormap definitions kept here (rather than in the pipeline) so a source
# fully owns the visual identity of its datasets.
_OISST_SST_CMAP: list[list[Any]] = [
    [0, "darkblue"],
    [20, "white"],
    [24, "yellow"],
    [28, "orange"],
    [30, "red"],
    [32, "darkred"],
    [35, "#470000"],
]
_OISST_ANOM_CMAP: list[list[Any]] = [
    [-3, "darkblue"],
    [-0.5, "lightblue"],
    [0, "white"],
    [1.5, "yellow"],
    [3, "red"],
    [5, "darkred"],
    [7, "#470000"],
]


class OisstSource(DataSource):
    id = "oisst"
    grid_shape = (720, 1440)

    datasets = {
        "sst": DatasetSpec(
            id="sst",
            cmap_def=_OISST_SST_CMAP,
            title_template="{date}\nSea Surface Temp, °C",
            equirect_basename="sst-temp",
            map_basename="sst-sst",
        ),
        "anom": DatasetSpec(
            id="anom",
            cmap_def=_OISST_ANOM_CMAP,
            title_template="{date}\nSea Surface Temp Variance from 1971–2000 Mean, °C",
            equirect_basename="sst-temp-anomaly",
            map_basename="sst-anom",
        ),
    }

    async def fetch(
        self,
        date: datetime.date,
        session: aiohttp.ClientSession,
        semaphore: Any,
    ) -> h5py.File:
        ym = f"{date.year:04}{date.month:02}"
        ymd = f"{date.year:04}{date.month:02}{date.day:02}"
        urls = [
            f"{_OISST_URL_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc",
            f"{_OISST_URL_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}_preliminary.nc",
        ]
        async with semaphore:
            print(f"Requesting OISST data for {date}...")
            try:
                data = await _try_fetch(urls, session)
            except DataFetchError:
                print(f"❌Failed to download {date} from any URL.")
                raise
            hdf = h5py.File(io.BytesIO(data), "r")
            print(f"✅Got hdf for {date}")
            return hdf

    def open_local(self, path: Path) -> AbstractContextManager[h5py.File]:
        @contextmanager
        def _opener() -> Iterator[h5py.File]:
            with h5py.File(path, "r") as hdf:
                yield hdf

        return _opener()

    def latlon_2d(self, raw: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        lat = raw["lat"][:]
        lon = raw["lon"][:]
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return lat_2d, lon_2d

    def equirect_filenames(self, dataset_id: str, date_str: str) -> list[str]:
        """OISST writes both a dated file and an undated "latest" copy.

        The undated copy preserves backward compatibility with the original
        S3 layout — older clients still fetch ``sst-temp-equirect.webp``
        without a date prefix.
        """
        spec = self.datasets[dataset_id]
        return [
            f"{date_str}-{spec.equirect_basename}-equirect.webp",
            f"{spec.equirect_basename}-equirect.webp",
        ]

    def get_data_array(
        self,
        raw: h5py.File,
        dataset_id: str,
        *,
        ice: bool = False,
        show: str = "default",
        lat_min: float = -90,
        lat_max: float = 90,
    ) -> np.ma.MaskedArray:
        """Return the masked, scaled 2D array for ``dataset_id``.

        ``show != "default"`` is a debug aid that returns the ice mask, the
        land mask, or the per-cell relative area instead of the dataset
        itself; it is only used by the ``--mode map --show …`` CLI path.
        """
        array = raw[dataset_id][0][0][:][:]
        scale_factor = raw[dataset_id].attrs["scale_factor"]
        assert raw[dataset_id].attrs["add_offset"] == 0

        lat_2d, lon_2d = self.latlon_2d(raw)
        ice_layer = raw["ice"][0][0][:][:] if ice else False
        ice_mask = ice_layer > 50
        # OISST flags land cells with the sentinel value -999.
        land_mask = array == -999
        lat_mask = np.logical_or(lat_2d < lat_min, lat_2d > lat_max)

        if show == "ice":
            return ice_layer  # type: ignore[return-value]
        if show == "land":
            return land_mask  # type: ignore[return-value]
        if show == "area":
            return _rel_area(lat_2d, lon_2d)  # type: ignore[return-value]

        masked = np.ma.array(
            array,
            mask=np.ma.mask_or(lat_mask, np.ma.mask_or(ice_mask, land_mask)),
        )
        return masked * scale_factor
