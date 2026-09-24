#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Render social-media preview cards (Open Graph images) from the time series.

For every (source, dataset, region) series in the exported time-series JSONs,
writes a 1200×630 JPEG to ``{out_dir}/{source}-{dataset}-{region}.jpg``: a globe
centered on the region showing the latest texture, the latest value with its
rank among the same calendar day in past years, and the year-overlay chart from
the site's Trends tab.

Also writes ``{out_dir}/manifest.json``, which globe-viz's Pages Function reads
to pick the card (and its title/description text) for each shared URL:

    {
      "updated": "2026-09-24T15:02:11+00:00",
      "cards": {
        "oisst-anom-nino_3_4": {
          "date": "2026-09-22",
          "title": "Niño 3.4 sea surface temp anomaly: +1.23 °C",
          "description": "...",
          "alt": "..."
        }
      }
    }

Card keys use the same source/dataset/region ids as the site's ``src``/``ds``/
``region`` URL params, so the function can build a key straight from the URL.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PIL import Image

import regions
from sources import SOURCES

BUCKET_URL = "https://climate-change-assets.s3.amazonaws.com/sea-surface-temp/"

# Rendered at 72 dpi so one point is one pixel: every size below is in pixels.
CARD_W, CARD_H = 1200, 630
DPI = 72

# Dark theme from globe-viz/src/styles/global.css, so a card reads as the site.
BG = "#0b1220"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
AXIS = "#6b7280"  # the site's translucent axis/grid whites, flattened onto BG
GRID = "#1d2535"
YEAR_OLD = "#2c3a5a"
YEAR_RECENT = "#9bc1ff"
YEAR_CURRENT = "#ff5050"
YEAR_PREV = "#ff9933"
YEAR_PREV2 = "#44bb77"
WARM = "#ff6b5b"
COOL = "#5ab0ff"
LAND = "#475569"

GLOBE_SIZE = 520
GLOBE_SUPERSAMPLE = 2
GLOBE_GLOW = 0.04  # atmosphere halo width, as a fraction of the globe radius
OUTSIDE_REGION_DIM = 0.6  # blend toward BG for cells outside the region

# How far back from a series' latest date to look for a texture: ERA5 textures
# can trail its time series by a few days.
TEXTURE_LOOKBACK_DAYS = 10

# Globe center (lat, lon) for each region, chosen to frame it with some context.
GLOBE_VIEWS: dict[str, tuple[float, float]] = {
    "global": (0, -110),
    "trop": (0, -150),
    "n_hemi": (35, -40),
    "s_hemi": (-35, 110),
    "nino_3_4": (0, -145),
    "pacific": (5, -160),
    "atlantic": (10, -35),
    "indian": (-15, 75),
    "arctic": (75, -20),
    "antarctic": (-65, 20),
}
DEFAULT_VIEW = (0, -150)

SOURCE_LABELS = {"oisst": "NOAA OISST", "era5": "ECMWF ERA5", "gfs": "NOAA GFS"}

# Mirrors DATASET_TITLE_FRAGMENT in globe-viz/src/components/Trends.tsx.
DATASET_LABELS = {
    "sst": "Sea surface temp",
    "anom": "Sea surface temp anomaly",
    "sst_anom": "Sea surface temp anomaly",
    "t2m": "Air temp",
    "t2m_anom": "Air temp anomaly",
    "t2m_mean": "Daily mean air temp",
    "t2m_max": "Daily max air temp",
    "t2m_min": "Daily min air temp",
    "t2m_mean_anom": "Daily mean air temp anomaly",
    "t2m_max_anom": "Daily max air temp anomaly",
    "t2m_min_anom": "Daily min air temp anomaly",
}
ANOMALY_BASELINE = "1971–2000"


@dataclass
class Series:
    region: str
    region_label: str
    dates: list[str]
    values: list[float]
    preliminary_from: str | None


# ---------------------------------------------------------------------------
# Globe
# ---------------------------------------------------------------------------


def region_grid(region_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    """In-region cells in texture (north-up) order, or None if it can't be known."""
    if region_id not in regions.REGIONS or shape != regions.GRID_SHAPE:
        return None
    h, w = shape
    lat = -90 + (np.arange(h) + 0.5) * 180 / h
    lon = (np.arange(w) + 0.5) * 360 / w
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    # Textures are written with origin="lower", so row 0 of the image is north.
    return np.flipud(regions.contains(region_id, lat_2d, lon_2d))


def render_globe(
    texture: np.ndarray, inside: np.ndarray | None, lat0: float, lon0: float
) -> np.ndarray:
    """Orthographic view of an equirect texture, composited over BG, as RGB floats."""
    n = GLOBE_SIZE * GLOBE_SUPERSAMPLE
    extent = 1 + GLOBE_GLOW
    y, x = np.mgrid[extent : -extent : n * 1j, -extent : extent : n * 1j]
    rho = np.maximum(np.hypot(x, y), 1e-12)
    on_disk = rho <= 1
    c = np.arcsin(np.minimum(rho, 1))
    phi0, lam0 = np.radians(lat0), np.radians(lon0)
    lat = np.arcsin(
        np.cos(c) * np.sin(phi0) + y * np.sin(c) * np.cos(phi0) / rho,
    )
    lon = lam0 + np.arctan2(
        x * np.sin(c), rho * np.cos(c) * np.cos(phi0) - y * np.sin(c) * np.sin(phi0)
    )

    h, w = texture.shape[:2]
    row = ((90 - np.degrees(lat)) / 180 * h).astype(int).clip(0, h - 1)
    col = ((np.degrees(lon) % 360) / 360 * w).astype(int).clip(0, w - 1)
    texel = texture[row, col] / 255.0
    rgb = texel[..., :3]
    if texture.shape[2] == 4:  # transparent cells are land (masked in the data)
        rgb = np.where(texel[..., 3:] > 0.5, rgb, to_rgb(LAND))
    bg = np.array(to_rgb(BG))
    if inside is not None:
        rgb = np.where(
            inside[row, col][..., None], rgb, rgb + (bg - rgb) * OUTSIDE_REGION_DIM
        )
    rgb = rgb * (0.55 + 0.45 * np.cos(c))[..., None]

    glow = np.clip(1 - (rho - 1) / GLOBE_GLOW, 0, 1) * 0.35
    halo = bg + (np.array(to_rgb(ACCENT)) - bg) * glow[..., None]
    img = np.where(on_disk[..., None], rgb, halo)
    s = GLOBE_SUPERSAMPLE
    return img.reshape(GLOBE_SIZE, s, GLOBE_SIZE, s, 3).mean(axis=(1, 3))


def load_texture(
    source_id: str, dataset_id: str, latest: date, maps_dir: Path
) -> np.ndarray | None:
    """Most recent texture on or before `latest`: local ./maps first, then S3."""
    source = SOURCES[source_id]()
    for back in range(TEXTURE_LOOKBACK_DAYS):
        day = (latest - timedelta(days=back)).isoformat()
        name = source.equirect_filenames(dataset_id, day)[0]
        local = maps_dir / name
        if local.exists():
            return np.asarray(Image.open(local))
        resp = requests.get(BUCKET_URL + name, timeout=30)
        if resp.ok:
            return np.asarray(Image.open(io.BytesIO(resp.content)))
    return None


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def leap_doy(d: date) -> int:
    """Day of year aligned to a leap year, as the site's Trends chart does."""
    return date(2000, d.month, d.day).timetuple().tm_yday - 1


def lerp_color(a: str, b: str, t: float) -> tuple[float, float, float]:
    ra, rb = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple(ra + (rb - ra) * t)


def by_year(series: Series) -> dict[int, tuple[list[int], list[float], int]]:
    """year → (x, y, prelim_index): NaN-broken at gaps, prelim_index -1 if none."""
    out: dict[int, tuple[list[int], list[float], int]] = {}
    prev: date | None = None
    for ds, v in zip(series.dates, series.values, strict=True):
        d = date.fromisoformat(ds)
        x, y, prelim = out.setdefault(d.year, ([], [], -1))
        if prev is not None and prev.year == d.year and (d - prev).days > 1:
            x.append(leap_doy(d))
            y.append(np.nan)
        if prelim < 0 and series.preliminary_from and ds >= series.preliminary_from:
            out[d.year] = (x, y, len(x))
        x.append(leap_doy(d))
        y.append(v)
        prev = d
    return out


def draw_chart(ax: plt.Axes, series: Series, anomaly: bool) -> None:
    years = by_year(series)
    first, last = min(years), max(years)
    span = max(1, last - first)
    styles = {
        last: (YEAR_CURRENT, 3.0),
        last - 1: (YEAR_PREV, 1.8),
        last - 2: (YEAR_PREV2, 1.5),
    }
    for year, (x, y, prelim) in sorted(years.items()):
        color, width = styles.get(
            year, (lerp_color(YEAR_OLD, YEAR_RECENT, (year - first) / span), 0.8)
        )
        z = 10 if year == last else 2
        if prelim < 0:
            ax.plot(x, y, color=color, linewidth=width, zorder=z)
        else:  # dotted from one point early, so the two halves join
            ax.plot(x[: prelim + 1], y[: prelim + 1], color=color, lw=width, zorder=z)
            ax.plot(x[prelim:], y[prelim:], color=color, lw=width, ls=":", zorder=z)
    x, y, _ = years[last]
    ax.plot(x[-1], y[-1], "o", ms=9, color=YEAR_CURRENT, mec=TEXT, mew=1.5, zorder=11)

    ax.set_facecolor(BG)
    ax.set_xlim(0, 365)
    month_starts = [leap_doy(date(2000, m, 1)) for m in (1, 4, 7, 10)]
    ax.set_xticks(month_starts, ["Jan", "Apr", "Jul", "Oct"])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    sign = "+" if anomaly else ""
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:{sign if round(v, 1) else ''}.1f}°")
    )
    ax.tick_params(colors=MUTED, labelsize=15, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(AXIS)

    # Legend: the three highlighted years, then the gradient's range.
    legend = [(str(y), styles[y][0]) for y in (last, last - 1, last - 2) if y in years]
    if first < last - 3:
        legend.append((f"{first}–{last - 3}", YEAR_RECENT))
    elif first == last - 3:
        legend.append((str(first), YEAR_RECENT))
    x_px = 0
    for label, color in legend:
        ax.annotate(
            f"● {label}",
            (0, 1),
            xycoords="axes fraction",
            xytext=(x_px, 10),
            textcoords="offset pixels",
            color=color,
            fontsize=15,
        )
        x_px += 22 + 10 * len(label)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10)
    return f"{n}{suffix or 'th'}"


def rank_phrase(series: Series) -> str:
    """e.g. "3rd warmest Sep 22 in 45 years on record"."""
    latest, value = series.dates[-1], series.values[-1]
    same_day = [v for d, v in zip(series.dates, series.values) if d[5:] == latest[5:]]
    warmer = sum(v > value for v in same_day)
    cooler = sum(v < value for v in same_day)
    word, rank = (
        ("warmest", warmer + 1) if warmer <= cooler else ("coolest", cooler + 1)
    )
    phrase = word if rank == 1 else f"{ordinal(rank)} {word}"
    d = date.fromisoformat(latest)
    return f"{phrase[0].upper()}{phrase[1:]} {d:%b} {d.day} in {len(same_day)} years on record"


def format_value(v: float, anomaly: bool) -> str:
    return f"{v:+.2f} °C" if anomaly else f"{v:.2f} °C"


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------


def fig_text(fig, x: float, y: float, s: str, **kw) -> None:
    """Place text at pixel (x, y) from the card's top-left corner."""
    fig.text(x / CARD_W, 1 - y / CARD_H, s, **kw)


def fig_axes(fig, x: float, y: float, w: float, h: float) -> plt.Axes:
    """Axes at a pixel rectangle measured from the card's top-left corner."""
    return fig.add_axes((x / CARD_W, 1 - (y + h) / CARD_H, w / CARD_W, h / CARD_H))


def render_card(
    source_id: str,
    dataset_id: str,
    series: Series,
    texture: np.ndarray,
    out_path: Path,
) -> dict[str, str]:
    """Write one card and return its manifest entry."""
    anomaly = dataset_id.endswith("anom")
    label = DATASET_LABELS[dataset_id]
    source_label = SOURCE_LABELS.get(source_id, source_id)
    region_name = series.region_label.split(" (")[0]
    value = series.values[-1]
    value_str = format_value(value, anomaly)
    latest = date.fromisoformat(series.dates[-1])
    rank = rank_phrase(series)

    fig = plt.figure(figsize=(CARD_W / DPI, CARD_H / DPI), dpi=DPI, facecolor=BG)

    lat0, lon0 = GLOBE_VIEWS.get(series.region, DEFAULT_VIEW)
    inside = region_grid(series.region, texture.shape[:2])
    globe = fig_axes(fig, 30, (CARD_H - GLOBE_SIZE) / 2, GLOBE_SIZE, GLOBE_SIZE)
    globe.imshow(render_globe(texture, inside, lat0, lon0), interpolation="none")
    globe.axis("off")

    left = 610
    fig_text(
        fig, left, 62, "CLIMATE DATA EXPLORER", color=ACCENT, fontsize=16, weight="bold"
    )
    fig_text(fig, left, 108, region_name, color=TEXT, fontsize=42, weight="bold")
    subtitle = f"{label} · {source_label}"
    if anomaly:
        subtitle = f"{label} vs. {ANOMALY_BASELINE} · {source_label}"
    fig_text(fig, left, 142, subtitle, color=MUTED, fontsize=19)
    value_color = (WARM if value > 0 else COOL) if anomaly else TEXT
    fig_text(fig, left, 222, value_str, color=value_color, fontsize=64, weight="bold")
    fig_text(
        fig,
        left,
        258,
        f"{rank} · {latest:%b} {latest.day}, {latest.year}",
        color=TEXT,
        fontsize=18,
    )

    draw_chart(fig_axes(fig, left + 52, 318, CARD_W - left - 92, 248), series, anomaly)
    fig_text(
        fig,
        CARD_W - 40,
        612,
        "globe-viz.oberbrunner.com",
        color=MUTED,
        fontsize=16,
        ha="right",
    )

    fig.savefig(
        out_path,
        dpi=DPI,
        facecolor=BG,
        pil_kwargs={"quality": 88, "optimize": True, "progressive": True},
    )
    plt.close(fig)

    lower = label.lower()
    first_year = series.dates[0][:4]
    return {
        "date": series.dates[-1],
        "title": f"{region_name} {lower}: {value_str}",
        "description": (
            f"{rank} ({source_label}, {latest:%b} {latest.day}, {latest.year}). "
            "Explore daily climate data on an interactive 3D globe with "
            "year-over-year charts."
        ),
        "alt": (
            f"Globe map of {lower}, {series.region_label}, beside a chart of "
            f"every year since {first_year} with {latest.year} in red."
        ),
    }


def render_dataset(
    source_id: str,
    dataset_id: str,
    all_series: list[Series],
    maps_dir: Path,
    out_dir: Path,
) -> dict[str, dict[str, str]]:
    """Render every region's card for one (source, dataset); one texture load."""
    latest = max(date.fromisoformat(s.dates[-1]) for s in all_series)
    texture = load_texture(source_id, dataset_id, latest, maps_dir)
    if texture is None:
        print(f"⚠️  {source_id}/{dataset_id}: no texture near {latest}; skipping")
        return {}
    entries = {}
    for series in all_series:
        key = f"{source_id}-{dataset_id}-{series.region}"
        entries[key] = render_card(
            source_id, dataset_id, series, texture, out_dir / f"{key}.jpg"
        )
        print(f"✓ {key}: {entries[key]['title']}")
    return entries


def collect_series(
    timeseries_dir: Path, patterns: list[str]
) -> dict[tuple[str, str], list[Series]]:
    """(source, dataset) → per-region series, for datasets the cards know."""
    grouped: dict[tuple[str, str], list[Series]] = {}
    for path in sorted(timeseries_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        for source_id, src in payload["sources"].items():
            if source_id not in SOURCES:
                continue
            for dataset_id, ds in src["datasets"].items():
                key = f"{source_id}-{dataset_id}-{payload['region']}"
                if dataset_id not in DATASET_LABELS or not ds["dates"]:
                    continue
                if patterns and not any(fnmatch.fnmatch(key, p) for p in patterns):
                    continue
                grouped.setdefault((source_id, dataset_id), []).append(
                    Series(
                        region=payload["region"],
                        region_label=payload.get("region_label", payload["region"]),
                        dates=ds["dates"],
                        values=ds["values"],
                        preliminary_from=src.get("preliminary_from"),
                    )
                )
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        default=Path("./maps/timeseries"),
        help="Per-region JSONs written by export_timeseries.py",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("./maps"),
        help="Where to look for this run's textures before falling back to S3",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./maps/og"))
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only render cards whose key matches, e.g. 'oisst-*-nino_3_4' "
        "(repeatable). The manifest then lists just those cards.",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    grouped = collect_series(args.timeseries_dir, args.only)
    if not grouped:
        print(f"❌ No time series found in {args.timeseries_dir}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cards: dict[str, dict[str, str]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(render_dataset, src, ds, series, args.maps_dir, args.out_dir)
            for (src, ds), series in sorted(grouped.items())
        ]
        for future in futures:
            cards.update(future.result())

    manifest = {
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "cards": dict(sorted(cards.items())),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"✓ {manifest_path}: {len(cards)} card(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
