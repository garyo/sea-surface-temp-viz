# SPDX-License-Identifier: MIT
"""Registry of pluggable climate-data sources.

Each entry maps a CLI/source id (e.g. ``"oisst"``) to a :class:`DataSource`
subclass. ``pipeline.py`` looks up the requested source here.
"""

from __future__ import annotations

from .base import DataSource, DatasetSpec
from .era5 import Era5Source
from .gfs import GfsSource
from .oisst import OisstSource

SOURCES: dict[str, type[DataSource]] = {
    OisstSource.id: OisstSource,
    Era5Source.id: Era5Source,
    GfsSource.id: GfsSource,
}

__all__ = [
    "DataSource",
    "DatasetSpec",
    "Era5Source",
    "GfsSource",
    "OisstSource",
    "SOURCES",
]
