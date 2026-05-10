# SPDX-License-Identifier: MIT
"""Region definitions and area-weighted aggregation for OISST-style 0.25° grids.

A region is either:
  - a bbox: dict with "lat" and "lon" tuples (inclusive ranges; lon in 0–360,
    matching the OISST `lon` coordinate convention).
  - a mask: dict with "mask" naming a .npy file under ./masks/ holding a boolean
    grid of shape GRID_SHAPE (True = in-region).

Bboxes are evaluated at runtime. Masks are precomputed by
scripts/precompute_masks.py and committed to the repo so the daily pipeline has
no shapefile dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# OISST is 720×1440 (0.25° lat × 0.25° lon, lon 0–360).
GRID_SHAPE = (720, 1440)

MASK_DIR = Path(__file__).parent / "masks"


REGIONS: dict[str, dict] = {
    # Global excludes polar latitudes — preserves the time-series semantics from
    # the pre-Phase-2 pipeline, which used lat_min=-60, lat_max=60 in
    # get_average_temp(). Changing this would shift every historical value.
    "global":    {"label": "Global (60°S–60°N)",
                  "bbox": {"lat": (-60, 60), "lon": (0, 360)}},
    "trop":      {"label": "Tropics (23.5°S–23.5°N)",
                  "bbox": {"lat": (-23.5, 23.5), "lon": (0, 360)}},
    "n_hemi":    {"label": "Northern Hemisphere (0–60°N)",
                  "bbox": {"lat": (0, 60), "lon": (0, 360)}},
    "s_hemi":    {"label": "Southern Hemisphere (60°S–0)",
                  "bbox": {"lat": (-60, 0), "lon": (0, 360)}},
    "nino_3_4":  {"label": "Niño 3.4 (5°S–5°N, 170°W–120°W)",
                  "bbox": {"lat": (-5, 5), "lon": (190, 240)}},
    "pacific":   {"label": "Pacific Ocean",   "mask": "pacific.npy"},
    "atlantic":  {"label": "Atlantic Ocean",  "mask": "atlantic.npy"},
    "indian":    {"label": "Indian Ocean",    "mask": "indian.npy"},
    "arctic":    {"label": "Arctic Ocean",    "mask": "arctic.npy"},
    "antarctic": {"label": "Southern Ocean",  "mask": "antarctic.npy"},
}


def region_ids() -> list[str]:
    return list(REGIONS.keys())


def label_for(region_id: str) -> str:
    return REGIONS[region_id]["label"]


_mask_cache: dict[str, np.ndarray] = {}


def _load_mask(filename: str) -> np.ndarray:
    cached = _mask_cache.get(filename)
    if cached is not None:
        return cached
    path = MASK_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Region mask not found: {path}. "
            f"Run scripts/precompute_masks.py to generate it."
        )
    mask = np.load(path)
    if mask.shape != GRID_SHAPE:
        raise ValueError(
            f"Mask {path} has shape {mask.shape}, expected {GRID_SHAPE}"
        )
    _mask_cache[filename] = mask
    return mask


def _bbox_exclude(lat_2d: np.ndarray, lon_2d: np.ndarray, bbox: dict) -> np.ndarray:
    """Boolean array marking cells OUTSIDE the bbox (True = excluded).

    Supports lon ranges that wrap the prime meridian (lon_lo > lon_hi).
    """
    lat_lo, lat_hi = bbox["lat"]
    lon_lo, lon_hi = bbox["lon"]
    out = (lat_2d < lat_lo) | (lat_2d > lat_hi)
    if lon_lo <= lon_hi:
        out |= (lon_2d < lon_lo) | (lon_2d > lon_hi)
    else:
        out |= (lon_2d < lon_lo) & (lon_2d > lon_hi)
    return out


def aggregate(
    data,
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    region_id: str,
) -> float:
    """Cosine-latitude-weighted average of `data` over `region_id`.

    `data` should already have land/ice/scale-factor masks applied by the caller
    (e.g. via get_processed_hdf_data_array with lat_min=-90, lat_max=90); this
    function adds the region's own lat/lon constraint on top. Cells excluded by
    the region get effective weight 0.

    Returns NaN if no cells are valid (fully masked).
    """
    region = REGIONS[region_id]
    if "bbox" in region:
        cell_excl = _bbox_exclude(lat_2d, lon_2d, region["bbox"])
    else:
        cell_excl = ~_load_mask(region["mask"])

    data_arr = np.ma.getdata(data)
    existing = np.ma.getmaskarray(data)
    combined = existing | cell_excl
    masked = np.ma.array(data_arr, mask=combined)
    weights = np.cos(np.radians(lat_2d))
    avg = np.ma.average(masked, weights=weights)
    return float(avg) if not np.ma.is_masked(avg) else float("nan")
