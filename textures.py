# SPDX-License-Identifier: MIT
"""Equirectangular globe textures: color-mapped images and 16-bit data textures.

Every texture date gets two renderings of the same 720×1440 field, north up:

- ``…-equirect.webp`` — the colormapped image older site versions display.
- ``…-data.webp`` — the values themselves, which the current globe decodes and
  colors in its shader (so it can re-baseline anomalies per pixel) and reads
  back exactly for the point readout.

**Data texture encoding** (``Rg16``; ``DATA_ENCODING`` for data textures,
written into each one's ``-metadata.json``): lossless RGB WebP holding a 16-bit count per
pixel, high byte in R and low byte in G (B is 0)::

    value_°C = offset + scale × (256·R + G)      count 0 = no data (land, ice)

With ``scale`` = 0.01 °C — OISST's own storage precision — and ``offset`` =
−200 °C, counts span −199.99…455.35 °C. Values are rounded to the nearest
0.01 °C; a field only uses as many of the 16 bits as its range needs, so the
constant high bits cost almost nothing after compression. There's no alpha
channel because browsers premultiply alpha when decoding images, which would
corrupt the counts of any partly transparent pixel.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

if TYPE_CHECKING:
    from sources.base import DatasetSpec, DataSource

NO_DATA = 0
MAX_COUNT = 0xFFFF


@dataclass(frozen=True)
class Rg16:
    """value = offset + scale × count, count in R (high byte) and G (low byte)."""

    scale: float
    offset: float
    format: str = "rg16"

    def as_metadata(self) -> dict:
        return asdict(self)

    def encode(self, values: np.ndarray) -> Image.Image:
        """Pack a (rows south→north) field into an image, north up.

        Masked or non-finite cells become count 0 (no data); values outside the
        representable range clamp to its ends.
        """
        data = np.ma.filled(np.ma.masked_invalid(values).astype(np.float64), np.nan)
        counts = np.rint((data - self.offset) / self.scale)
        counts = np.where(np.isfinite(counts), np.clip(counts, 1, MAX_COUNT), NO_DATA)
        counts = counts.astype(np.uint16)[::-1]
        rgb = np.zeros((*counts.shape, 3), dtype=np.uint8)
        rgb[..., 0] = counts >> 8
        rgb[..., 1] = counts & 0xFF
        return Image.fromarray(rgb, "RGB")

    def decode(self, image: Image.Image) -> np.ma.MaskedArray:
        """Inverse of encode (rows back to south→north)."""
        rgb = np.asarray(image.convert("RGB"))[::-1]
        counts = rgb[..., 0].astype(np.int32) * 256 + rgb[..., 1]
        return np.ma.masked_equal(counts, NO_DATA) * self.scale + self.offset


DATA_ENCODING = Rg16(scale=0.01, offset=-200.0)


def save_lossless_webp(image: Image.Image, path: Path) -> None:
    # method 4 is ~3× faster than 6 for <2% larger files.
    image.save(path, "WEBP", lossless=True, quality=100, method=4)


def rescale(x, oldmin, oldmax, newmin, newmax):
    return ((x - oldmin) / (oldmax - oldmin)) * (newmax - newmin) + newmin


def rescale_colormap_def_to_01(cmap):
    """Map the leftmost x-value to 0 and the rightmost to 1, preserving colors."""
    vmin = cmap[0][0]
    vmax = cmap[-1][0]
    return [[rescale(x[0], vmin, vmax, 0, 1), *x[1:]] for x in cmap]


def colormap_for(spec: DatasetSpec) -> tuple[LinearSegmentedColormap, float, float]:
    cmap = LinearSegmentedColormap.from_list(
        f"{spec.id}_cmap", rescale_colormap_def_to_01(spec.cmap_def)
    )
    return cmap, spec.cmap_def[0][0], spec.cmap_def[-1][0]


def texture_metadata(spec: DatasetSpec, date: datetime.date) -> dict:
    date_str = date.isoformat()
    metadata = {
        "cmap": spec.cmap_def,
        "title": spec.title_template.format(date=date_str),
        "dataset": spec.id,
        "date": date_str,
        "year": date.year,
        "month": date.month,
        "day": date.day,
    }
    # Orthogonal axes (variable / statistic / kind) so the frontend can render
    # independent selectors — e.g. a GFS min/mean/max button group. Only
    # emitted for datasets that declare them.
    for axis in ("variable", "statistic", "kind"):
        val = getattr(spec, axis)
        if val is not None:
            metadata[axis] = val
    return metadata


def write_metadata(metadata: dict, texture_path: Path) -> None:
    out = texture_path.parent / (texture_path.stem + "-metadata.json")
    out.write_text(json.dumps(metadata))


def write_textures(
    source: DataSource,
    dataset_id: str,
    date: datetime.date,
    data: np.ma.MaskedArray,
    out_dir: Path,
) -> None:
    """Write one date's color-mapped and data textures, each with metadata."""
    spec = source.datasets[dataset_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = texture_metadata(spec, date)

    cmap, vmin, vmax = colormap_for(spec)
    for filename in source.equirect_filenames(dataset_id, date.isoformat()):
        path = out_dir / filename
        plt.imsave(path, data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        write_metadata(metadata, path)

    path = out_dir / source.data_texture_filename(dataset_id, date.isoformat())
    save_lossless_webp(DATA_ENCODING.encode(data), path)
    write_metadata({**metadata, "encoding": DATA_ENCODING.as_metadata()}, path)
