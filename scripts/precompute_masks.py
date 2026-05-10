#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "regionmask", "xarray"]
# ///
# SPDX-License-Identifier: MIT
"""Precompute ocean-basin boolean masks on the OISST 0.25° grid.

Uses the IPCC AR6 reference ocean regions (15 polygons) via regionmask, which
provide clean basin-scale boundaries (Natural Earth's marine_polys is too
fine-grained — its parent "INDIAN OCEAN" polygon is overridden by sub-seas
like Arabian Sea via last-match-wins rasterization).

Mapping to our region IDs:
  - atlantic  = NAO + EAO + SAO + CAR
  - pacific   = NPO + EPO + SPO
  - indian    = ARS + BOB + EIO + SIO
  - arctic    = ARO
  - antarctic = SOO
Mediterranean (MED) and S.E.Asia (SEA) are intentionally excluded — they sit
between basins and don't belong unambiguously to any one of the above.

Generated .npy files are committed to the repo so the daily pipeline has no
shapefile/regionmask dependency.

Usage:
    uv run scripts/precompute_masks.py [--out-dir ./masks]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import regionmask
import xarray as xr

BASIN_GROUPS: dict[str, list[str]] = {
    "atlantic":  ["NAO", "EAO", "SAO", "CAR"],
    "pacific":   ["NPO", "EPO", "SPO"],
    "indian":    ["ARS", "BOB", "EIO", "SIO"],
    "arctic":    ["ARO"],
    "antarctic": ["SOO"],
}


def oisst_coords() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) 1D coordinate arrays matching the OISST 0.25° grid."""
    lat = np.arange(-89.875, 90.0, 0.25)
    lon = np.arange(0.125, 360.0, 0.25)
    assert lat.size == 720, f"lat size {lat.size} != 720"
    assert lon.size == 1440, f"lon size {lon.size} != 1440"
    return lat, lon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "masks",
        help="Where to write basin .npy files",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lat, lon = oisst_coords()
    regions = regionmask.defined_regions.ar6.ocean

    abbrev_to_number: dict[str, int] = dict(zip(regions.abbrevs, regions.numbers))
    print(f"Loaded {len(abbrev_to_number)} AR6 ocean regions")

    print("Rasterizing onto OISST grid (720×1440)...")
    coords = xr.Dataset(coords={"lat": lat, "lon": lon})
    mask_da = regions.mask(coords)
    mask_arr = mask_da.values  # shape (720, 1440), float with NaN outside regions

    for region_id, abbrevs in BASIN_GROUPS.items():
        try:
            numbers = [abbrev_to_number[a] for a in abbrevs]
        except KeyError as e:
            print(f"❌ AR6 region not found: {e}", file=sys.stderr)
            print("Available:", sorted(abbrev_to_number.keys()), file=sys.stderr)
            return 1

        bool_mask = np.isin(mask_arr, numbers)
        out_path = args.out_dir / f"{region_id}.npy"
        np.save(out_path, bool_mask)
        coverage = bool_mask.sum() / bool_mask.size * 100
        print(f"✓ {region_id}.npy ({coverage:.1f}% of grid) → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
