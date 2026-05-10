#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "regionmask", "xarray"]
# ///
# SPDX-License-Identifier: MIT
"""Precompute ocean-basin boolean masks on the OISST 0.25° grid.

Every basin is the union of explicit Natural Earth `ocean_basins_50`
polygons. We use Natural Earth instead of AR6 reference regions because
AR6 leaves large gaps in marginal seas (Bering, Sea of Japan, South China
Sea, Norwegian, Barents, …); the Natural Earth set covers them all.

Definition recipe (loosely IHO "Limits of Oceans and Seas" minus the
semi-enclosed bodies whose climate is distinct enough to bias the basin
mean):

  - INCLUDE marginal seas that share the basin's open-ocean dynamics
    (Bering, Sea of Japan, Caribbean, Norwegian, Barents, Kara, etc.).
  - EXCLUDE semi-enclosed / brackish / climate-distinct bodies:
    Mediterranean (+ Adriatic/Aegean/Ionian/Tyrrhenian/Balearic/Lion),
    Baltic (+ Bothnia/Finland), Black, Hudson (+ Strait/James/Ungava),
    Red Sea, Persian Gulf, Caspian. Rivers, Inner Sea, Inner Seas, and
    Caspian are skipped (not open ocean).

regionmask's last-match-wins rasterization makes np.isin necessary
rather than naming just the parent ocean polygons.

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

# Natural Earth polygon names per basin. Each list is the union of polygons
# whose cells should belong to that basin's mask. See module docstring for
# the inclusion recipe.
NE_BASIN_GROUPS: dict[str, list[str]] = {
    "pacific": [
        "North Pacific Ocean", "South Pacific Ocean",
        # Western Pacific marginal seas
        "Bering Sea", "Sea of Okhotsk", "Sea of Japan", "Yellow Sea", "Bo Hai",
        "East China Sea", "South China Sea", "Philippine Sea",
        "Korea Strait", "Taiwan Strait", "Luzon Strait",
        "Gulf of Thailand", "Gulf of Tonkin",
        # Indonesian / Southeast Asian seas
        "Sulu Sea", "Celebes Sea", "Banda Sea", "Ceram Sea", "Molucca Sea",
        "Makassar Strait", "Java Sea",
        # Australasia (Pacific side per IHO 2002)
        "Coral Sea", "Tasman Sea", "Bay of Plenty",
        "Solomon Sea", "Bismarck Sea",
        "Arafura Sea", "Gulf of Carpentaria", "Great Barrier Reef",
        # American Pacific side
        "Gulf of Alaska", "Cook Inlet", "Bristol Bay", "Shelikhova Gulf",
        "Golfo de California", "Golfo de Panamá",
    ],
    "atlantic": [
        "North Atlantic Ocean", "South Atlantic Ocean",
        # American Atlantic side
        "Caribbean Sea", "Gulf of Mexico", "Bahía de Campeche",
        "Straits of Florida", "Gulf of Honduras",
        "Bay of Fundy", "Gulf of Maine", "Gulf of Saint Lawrence",
        "Chesapeake Bay", "Sargasso Sea",
        # NE Atlantic + European seas (excl. Mediterranean + Baltic)
        "Bay of Biscay", "North Sea", "English Channel", "Bristol Channel",
        "Irish Sea",
        "Norwegian Sea", "Greenland Sea",
        # NW Atlantic / Labrador (Baffin Bay goes to Arctic)
        "Labrador Sea", "Davis Strait",
        # African Atlantic side
        "Gulf of Guinea",
        # SW Atlantic
        "Golfo San Jorge", "Río de la Plata",
        # Strait of Gibraltar is the open-ocean side of the Med opening
        "Strait of Gibraltar",
    ],
    "arctic": [
        "Arctic Ocean",
        "Barents Sea", "Kara Sea", "Laptev Sea", "Chukchi Sea", "Beaufort Sea",
        "White Sea",
        "Baffin Bay", "Melville Bay",
        "Amundsen Gulf", "Viscount Melville Sound", "The North Western Passages",
    ],
    "antarctic": [
        "SOUTHERN OCEAN",
        "Ross Sea", "Weddell Sea", "Amundsen Sea", "Bellingshausen Sea",
        # Drake Passage + Scotia Sea: south of the polar front, included here
        # rather than in Atlantic (IHO 2002 Southern Ocean extends to 60°S).
        "Drake Passage", "Scotia Sea",
    ],
    "indian": [
        "INDIAN OCEAN",
        "Arabian Sea", "Bay of Bengal", "Andaman Sea", "Laccadive Sea",
        "Mozambique Channel",
        # Open extensions of the Arabian Sea — included; Red Sea / Persian
        # Gulf are excluded as semi-enclosed (very different climate).
        "Gulf of Aden", "Gulf of Oman",
        "Gulf of Mannar", "Gulf of Kutch",
        "Strait of Malacca", "Strait of Singapore",
        "Timor Sea", "Great Australian Bight",
    ],
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
    coords = xr.Dataset(coords={"lat": lat, "lon": lon})

    ne = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
    ne_name_to_number: dict[str, int] = dict(zip(ne.names, ne.numbers))
    print(f"Loaded {len(ne_name_to_number)} Natural Earth ocean polygons")

    print("Rasterizing Natural Earth polygons onto OISST grid (720×1440)...")
    ne_mask_arr = ne.mask(coords).values

    for region_id, names in NE_BASIN_GROUPS.items():
        try:
            numbers = [ne_name_to_number[n] for n in names]
        except KeyError as e:
            print(f"❌ Natural Earth polygon not found: {e}", file=sys.stderr)
            print("Available:", sorted(ne_name_to_number.keys()), file=sys.stderr)
            return 1
        bool_mask = np.isin(ne_mask_arr, numbers)
        out_path = args.out_dir / f"{region_id}.npy"
        np.save(out_path, bool_mask)
        coverage = bool_mask.sum() / bool_mask.size * 100
        print(f"✓ {region_id}.npy ({coverage:.1f}% of grid) → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
