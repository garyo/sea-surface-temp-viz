# SPDX-License-Identifier: MIT
"""DataSource ABC: contract every climate-data source must satisfy.

A *source* (OISST, ERA5, MODIS LST, …) owns the details of fetching one day
of data, extracting per-cell arrays, and describing how each of its datasets
should be visualized. The pipeline orchestrates I/O, caching, and plotting on
top of this contract — sources never touch matplotlib or the cache file.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import regions


@dataclass(frozen=True)
class DatasetSpec:
    """Display + file-naming metadata for one dataset within a source.

    ``cmap_def`` is a list of ``[value, color]`` pairs (the same shape used by
    ``LinearSegmentedColormap.from_list`` after rescaling). ``title_template``
    is a format string with a ``{date}`` placeholder. ``equirect_basename`` is
    the stem inserted between date and ``-equirect.webp`` — e.g.
    ``"sst-temp"`` produces ``"2026-05-08-sst-temp-equirect.webp"``.
    """

    id: str
    cmap_def: list[list[Any]]
    title_template: str
    equirect_basename: str
    map_basename: str


class DataSource(ABC):
    """Contract for a climate-data source.

    Subclasses are stateless apart from class-level configuration (``id``,
    ``grid_shape``, ``datasets``). Each call to :meth:`fetch` (or
    :meth:`open_local`) returns a fresh ``raw`` object that the rest of the
    methods consume.
    """

    id: str
    grid_shape: tuple[int, int]
    datasets: dict[str, DatasetSpec]

    @abstractmethod
    async def fetch(
        self,
        date: datetime.date,
        session: Any,
        semaphore: Any,
    ) -> Any:
        """Download one day of source data and return a raw handle."""

    @abstractmethod
    def open_local(self, path: Path) -> AbstractContextManager[Any]:
        """Open a previously-downloaded file as a raw handle (context manager)."""

    @abstractmethod
    def get_data_array(
        self,
        raw: Any,
        dataset_id: str,
        **kwargs: Any,
    ) -> np.ma.MaskedArray:
        """Return a 2D masked array for ``dataset_id`` (lat × lon, native grid).

        Sources may accept extra ``**kwargs`` for source-specific knobs (e.g.
        OISST's ``ice``, ``show``, ``lat_min``, ``lat_max``). The default
        aggregation path calls this with no extras and expects the full grid
        with land/invalid cells masked out.
        """

    @abstractmethod
    def latlon_2d(self, raw: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lat_2d, lon_2d)`` meshgrids matching the data array shape."""

    def aggregate_all_regions(self, raw: Any) -> dict[str, dict[str, float]]:
        """Compute area-weighted means for every (dataset, region) combo.

        Returns ``{dataset_id: {region_id: value}}``. Subclasses can override
        for performance, but the default fits any source whose datasets share
        the same lat/lon grid.
        """
        lat_2d, lon_2d = self.latlon_2d(raw)
        result: dict[str, dict[str, float]] = {}
        for ds_id in self.datasets:
            data = self.get_data_array(raw, ds_id)
            result[ds_id] = {
                rid: regions.aggregate(data, lat_2d, lon_2d, rid)
                for rid in regions.region_ids()
            }
        return result

    def equirect_filenames(self, dataset_id: str, date_str: str) -> list[str]:
        """Filenames to write for an equirect texture.

        Default: a single dated filename. Subclasses can return additional
        legacy/unprefixed names (OISST does this so old un-dated S3 links
        keep working).
        """
        spec = self.datasets[dataset_id]
        return [f"{date_str}-{spec.equirect_basename}-equirect.webp"]

    def map_filename(self, dataset_id: str) -> str:
        """Filename for a mollweide map plot."""
        spec = self.datasets[dataset_id]
        return f"{spec.map_basename}-map.webp"
