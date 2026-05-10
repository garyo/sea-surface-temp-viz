#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib", "cartopy", "scipy"]
# ///
# SPDX-License-Identifier: MIT
"""Plot every region defined in regions.py onto a world map, so basin masks
and bbox extents can be eyeballed for coverage gaps.

Run:
    uv run scripts/visualize_regions.py [--out /tmp/regions-map.png]

Writes a single PNG. Ocean basin masks are filled with distinct colors;
bbox regions are drawn as outlined rectangles (Niño 3.4 filled too — it's
small enough to read).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import regions  # noqa: E402


# Order matters for the label image — later regions overwrite earlier ones if
# masks overlap (in practice they don't with AR6).
BASIN_ORDER = ['pacific', 'atlantic', 'indian', 'arctic', 'antarctic']
BASIN_COLORS = {
    'pacific':   '#1f77b4',
    'atlantic':  '#2ca02c',
    'indian':    '#ff7f0e',
    'arctic':    '#9467bd',
    'antarctic': '#d62728',
}

# bbox region styling
BBOX_STYLE = {
    'global':   {'edge': '#000000', 'face': 'none',      'lw': 1.6, 'ls': '-'},
    'trop':     {'edge': '#b8860b', 'face': '#fff7c0',   'lw': 1.0, 'ls': '--', 'alpha': 0.30},
    'n_hemi':   {'edge': '#1565c0', 'face': 'none',      'lw': 0.8, 'ls': ':'},
    's_hemi':   {'edge': '#6a1b9a', 'face': 'none',      'lw': 0.8, 'ls': ':'},
    'nino_3_4': {'edge': '#b71c1c', 'face': '#ef9a9a',   'lw': 1.6, 'ls': '-', 'alpha': 0.55},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/tmp/regions-map.png'))
    parser.add_argument(
        '--mask-dir', type=Path,
        default=Path(__file__).resolve().parent.parent / 'masks',
    )
    args = parser.parse_args(argv)

    fig = plt.figure(figsize=(15, 7.5))
    # PlateCarree centered on -100° so the Americas / Pacific / Atlantic split
    # matches the globe-viz default camera (Mexico-centered).
    proj = ccrs.PlateCarree(central_longitude=-100)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()

    # Build a single integer label image: 0 = no basin, 1..N for each basin.
    labels = np.zeros(regions.GRID_SHAPE, dtype=np.int8)
    for i, region_id in enumerate(BASIN_ORDER, start=1):
        mask = np.load(args.mask_dir / f'{region_id}.npy')
        labels[mask] = i

    # ListedColormap: index 0 → transparent, 1..N → basin colors.
    base_colors = ['#00000000']  # fully transparent for 'no basin'
    base_colors += [BASIN_COLORS[r] for r in BASIN_ORDER]
    cmap = ListedColormap(base_colors)

    # OISST lon runs 0..360; cartopy expects -180..180 in PlateCarree. Roll
    # the array so longitudes line up after we declare extent=[-180, 180].
    labels_180 = np.roll(labels, labels.shape[1] // 2, axis=1)

    ax.imshow(
        labels_180,
        extent=[-180, 180, -90, 90],
        origin='lower',
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0,
        vmax=len(BASIN_ORDER),
        interpolation='nearest',
        alpha=0.6,
    )

    # Coastlines + land outlines on top so the user can see the polygon edges.
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.5, color='#444', zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.4, zorder=1)

    # bbox overlays. OISST uses lon 0..360, cartopy uses -180..180; convert.
    def lon360_to_180(x: float) -> float:
        return x - 360 if x > 180 else x

    for region_id, style in BBOX_STYLE.items():
        bbox = regions.REGIONS[region_id]['bbox']
        lat_lo, lat_hi = bbox['lat']
        lon_lo_360, lon_hi_360 = bbox['lon']
        lon_lo = lon360_to_180(lon_lo_360)
        lon_hi = lon360_to_180(lon_hi_360)
        # If the box wraps the dateline, split into two rectangles.
        if lon_lo <= lon_hi:
            spans = [(lon_lo, lon_hi)]
        else:
            spans = [(lon_lo, 180), (-180, lon_hi)]
        for x0, x1 in spans:
            rect = Rectangle(
                (x0, lat_lo), x1 - x0, lat_hi - lat_lo,
                edgecolor=style['edge'],
                facecolor=style['face'],
                linewidth=style['lw'],
                linestyle=style['ls'],
                alpha=style.get('alpha', 1.0),
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
            ax.add_patch(rect)

    # Legend: ocean basins as filled patches, bbox regions as outline patches.
    legend_patches = []
    for r in BASIN_ORDER:
        legend_patches.append(Patch(
            facecolor=BASIN_COLORS[r], alpha=0.6,
            edgecolor='none', label=regions.label_for(r),
        ))
    for r, style in BBOX_STYLE.items():
        legend_patches.append(Patch(
            facecolor=style['face'] if style['face'] != 'none' else 'white',
            edgecolor=style['edge'],
            linewidth=style['lw'],
            linestyle=style['ls'],
            alpha=style.get('alpha', 1.0) if style['face'] != 'none' else 1.0,
            label=regions.label_for(r),
        ))
    ax.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        'Region definitions (AR6 ocean basins as masks; '
        'bboxes for global/tropics/hemispheres/Niño 3.4)',
        fontsize=11,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'✓ Wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
