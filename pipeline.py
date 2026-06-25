# SPDX-License-Identifier: MIT
"""CLI entry point for the climate-data pipeline.

Selects a source via ``--source``, then dispatches to one of three modes:

* ``--mode graph`` — multi-year line plot from cached aggregates (cache-misses
  trigger a fetch + full-region aggregation).
* ``--mode map`` — mollweide global map for a single date (PNG/WebP).
* ``--mode texture`` — equirectangular texture for the 3D globe (WebP +
  metadata JSON).

The source object owns data fetching, extraction, and the per-dataset
visual identity (cmap, title, filenames). This module owns I/O — the
on-disk cache, matplotlib calls, and CLI plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import datetime
import json
import pathlib
import sys
from collections import defaultdict
from typing import DefaultDict

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import urllib3
from matplotlib.colors import LinearSegmentedColormap

from sources import SOURCES, DataSource
from sources.base import DatasetSpec
from sources.oisst import DataFetchError

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

n_concurrent_requests = 20
dpi = 150


def rescale(x, oldmin, oldmax, newmin, newmax):
    return ((x - oldmin) / (oldmax - oldmin)) * (newmax - newmin) + newmin


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if type(o) is np.float32:
            return o.item()
        return json.JSONEncoder.default(self, o)


# ---------------------------------------------------------------------------
# Disk cache. Key shape: ``YYYY-MM-DD-{source}-{dataset}-{region}``.
# scripts/migrate_cache.py produced this layout from the legacy schema; both
# pipeline.py and export_timeseries.py treat the cache as the source of truth.
# ---------------------------------------------------------------------------

temps_cache_file: str = "./data-cache.json"
temps_cache: dict = {}


def cache_key(year: int, mo: int, day: int, source: str, dataset: str, region: str) -> str:
    return f"{year:04}-{mo:02}-{day:02}-{source}-{dataset}-{region}"


def save_cache() -> None:
    json_data = json.dumps(temps_cache, sort_keys=True, indent=2, cls=NumpyArrayEncoder)
    with open(temps_cache_file, "w") as f:
        f.write(json_data)


def load_cache(path) -> None:
    global temps_cache_file
    global temps_cache
    temps_cache_file = str(path)
    try:
        with open(temps_cache_file, "r") as f:
            temps_cache = json.load(f)
    except IOError:
        temps_cache = {}


# ---------------------------------------------------------------------------
# Fetch + aggregate
# ---------------------------------------------------------------------------

async def get_temp_for_date(
    source: DataSource,
    year: int,
    mo: int,
    day: int,
    dataset_name: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    region: str = "global",
):
    """Return ``(year, mo, day, value)`` for one date, populating the cache.

    On cache miss, fetches the source for that date and writes ALL
    (dataset × region) aggregates back to the cache — the download dominates
    cost, so we never want to revisit the same date for a different region.
    """
    async with lock:
        cached = temps_cache.get(cache_key(year, mo, day, source.id, dataset_name, region))
        if cached is not None:
            return (year, mo, day, cached)

    try:
        date = datetime.date(year, mo, day)
        raw = await source.fetch(date, session, semaphore)
        all_aggs = source.aggregate_all_regions(raw)
        for ds_name, by_region in all_aggs.items():
            for region_id, val in by_region.items():
                temps_cache[cache_key(year, mo, day, source.id, ds_name, region_id)] = val
        t = temps_cache[cache_key(year, mo, day, source.id, dataset_name, region)]
        print(f"Computed {dataset_name} {year}-{mo:02}-{day:02} ({region}): {t:.4f}°C")
        do_save = True
    except DataFetchError:
        t = np.nan
        do_save = False

    if do_save:
        async with lock:
            save_cache()
    return (year, mo, day, t)


# ---------------------------------------------------------------------------
# Plotting helpers (source-agnostic)
# ---------------------------------------------------------------------------

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


def plot_globe_dataset(data, lat_1d, lon_1d, vmin, vmax, cmap, title):
    """Mollweide projection of one date's data."""
    _, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "mollweide"})
    ax.grid(visible=True)

    # mollweide expects lon ∈ [-π, π] and lat ∈ [-π/2, π/2]; OISST stores
    # lon ∈ [0, 360], so roll the array around the longitude axis.
    data_for_map = np.roll(data, data.shape[1] // 2, axis=1)
    lon_for_map = lon_1d - 180
    c = ax.pcolormesh(
        np.radians(lon_for_map),
        np.radians(lat_1d),
        data_for_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(c, orientation="horizontal", pad=0.05)
    plt.title(title)
    plt.suptitle(
        f"Created at {datetime.datetime.now()}\n"
        f"Copyright {datetime.date.today().year} Gary Oberbrunner",
        fontsize=7,
        y=0.97,
    )
    plt.tight_layout()
    return plt


def plot_equirect_dataset(data, vmin, vmax, cmap, outfile):
    plt.imsave(outfile, data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")


def save_metadata(metadata, outfile) -> None:
    p = pathlib.Path(outfile)
    out = p.parent / (p.stem + "-metadata.json")
    out.write_text(json.dumps(metadata))


# ---------------------------------------------------------------------------
# Map / texture mode
# ---------------------------------------------------------------------------

async def process_map(source: DataSource, args) -> None:
    if args.days_ago:
        date = datetime.date.today() - datetime.timedelta(days=args.days_ago)
    else:
        date = datetime.date(args.year, args.month, args.day)
    date_str = date.isoformat()

    semaphore = asyncio.Semaphore(n_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        try:
            raw = await source.fetch(date, session, semaphore)
        except DataFetchError:
            print(f"❌Failed to get {source.id} data for {date}")
            raise

    spec = source.datasets[args.dataset]
    data = source.get_data_array(
        raw, args.dataset, ice=args.ice, show=args.show
    )
    cmap, vmin, vmax = colormap_for(spec)
    title = spec.title_template.format(date=date_str)

    if args.mode == "map":
        # OISST raw still exposes lat/lon as h5py datasets; pull the 1D arrays
        # for the projection. (Sources with non-h5py raw can override via a
        # render_map method later if needed.)
        lat_1d = raw["lat"][:]
        lon_1d = raw["lon"][:]
        plot_globe_dataset(data, lat_1d, lon_1d, vmin, vmax, cmap, title)
        if args.out:
            out_dir = pathlib.Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / source.map_filename(args.dataset), dpi=dpi)
        else:
            plt.show()
        return

    # texture mode
    metadata = {
        "cmap": spec.cmap_def,
        "title": title,
        "dataset": args.dataset,
        "date": date_str,
        "year": date.year,
        "month": date.month,
        "day": date.day,
    }
    # Orthogonal axes (variable / statistic / kind) so the frontend can render
    # independent selectors — e.g. a GFS min/mean/max button group. Only emitted
    # for datasets that declare them; legacy single-axis datasets omit them.
    for axis in ("variable", "statistic", "kind"):
        val = getattr(spec, axis)
        if val is not None:
            metadata[axis] = val
    if args.out:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in source.equirect_filenames(args.dataset, date_str):
            out_path = out_dir / filename
            plot_equirect_dataset(data, vmin, vmax, cmap, out_path)
            save_metadata(metadata, out_path)


# ---------------------------------------------------------------------------
# Graph mode (year-overlay line plot)
# ---------------------------------------------------------------------------

def create_cache_dict() -> DefaultDict[int, DefaultDict[int, DefaultDict[int, float]]]:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(float)))


async def process_all(source: DataSource, args) -> None:
    async def get_data(dataset_name, session, semaphore, lock):
        start_year = args.start_year
        end_year = datetime.date.today().year
        tasks = []
        for year in range(start_year, end_year + 1):
            for mo in range(1, 13):
                num_days = calendar.monthrange(year, mo)[1]
                for day in range(1, num_days + 1):
                    if datetime.date(year, mo, day) > datetime.date.today():
                        continue
                    tasks.append(
                        asyncio.create_task(
                            get_temp_for_date(
                                source, year, mo, day, dataset_name,
                                session, semaphore, lock,
                            )
                        )
                    )
        results = create_cache_dict()
        for task in asyncio.as_completed(tasks):
            year, mo, day, val = await task
            async with lock:
                results[year][mo][day] = val
        return results

    year_cmap = LinearSegmentedColormap.from_list("year_cmap", ["lightgray", "darkblue"])

    def year_day_to_date(year, day):
        return datetime.date(year, 1, 1) + datetime.timedelta(days=day)

    # Align all post-Feb days across years by pretending each is in a leap year.
    def ymd_to_year_day_for_graph(year, mo, day):
        while not calendar.isleap(year):
            year += 1
        return datetime.date(year, mo, day).timetuple().tm_yday - 1

    def plot_fig(temps, title, use_ice_mask):
        _, ax = plt.subplots(figsize=(14, 8))
        years = np.array(sorted(list(temps.keys())))
        record = [-10000, (0, 0, 0)]

        def years_ago(n):
            if n >= len(years):
                return years[0]
            return years[-(n + 1)]

        for year in years:
            x, y = [], []
            for month in np.array(sorted(list(temps[year].keys()))):
                for day in np.array(sorted(list(temps[year][month].keys()))):
                    val = temps[year][month][day]
                    if not np.isnan(val):
                        if val > record[0]:
                            record[0] = val
                            record[1] = (year, month, day)
                        x.append(ymd_to_year_day_for_graph(year, month, day))
                        y.append(val)

            cmap_index = (year - years[0]) / (years[-1] - years[0])
            linewidth = 0.5
            color = year_cmap(cmap_index)
            if year == years_ago(0):
                color, linewidth = "red", 2
            if year == years_ago(1):
                color, linewidth = "orange", 1.5
            if year == years_ago(2):
                color = "green"
            label = f"{year}" if (year > years_ago(4) or year < years[2]) else None
            ax.plot(x, y, label=label, color=color, linewidth=linewidth)
            if year == years[-1] and len(x) > 0:
                last_date = year_day_to_date(year, x[-1])
                ax.annotate(
                    f"{last_date}\n{y[-1]:.2f}°C",
                    xy=(x[-1], y[-1]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    verticalalignment="top",
                    horizontalalignment="left",
                    color="blue",
                )
                ax.plot(x[-1], y[-1], marker=".", markersize=5, color="blue")
        if record[0] > -10000:
            record_val = record[0]
            record_x = ymd_to_year_day_for_graph(*record[1])
            ax.axhline(y=record_val, color="gray", linewidth=0.5, linestyle="dashed")
            ax.plot(record_x, record_val, marker=".", markersize=5, color="black")
            ax.annotate(
                f"record: {datetime.date(*record[1])}={record_val:.3f}",
                xy=(record_x, record_val),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=6,
            )
        plt.title(title)
        plt.suptitle(
            f"Years: {years[0]}-{years[-1]}. Created {datetime.datetime.now()}\n"
            f"Copyright {datetime.date.today().year} Gary Oberbrunner",
            fontsize=7,
            y=0.97,
        )
        plt.legend(loc="lower right")
        plt.xticks([])
        plt.grid(axis="y")
        ice_msg = ", ignoring ice > 50%" if use_ice_mask else ""
        msg = f"""
        Data from www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr
        All samples weighted by grid size, 60°N to 60°S{ice_msg}.
        See https://www.ncei.noaa.gov/products/climate-data-records/sea-surface-temperature-optimum-interpolation
        """
        plt.text(0, 0, msg, ha="left", va="top", transform=plt.gca().transAxes, fontsize=9)
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=dpi)
        else:
            plt.show()

    semaphore = asyncio.Semaphore(n_concurrent_requests)
    lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        temp_data_by_date = await get_data(args.dataset, session, semaphore, lock)

    plot_fig(
        temp_data_by_date,
        (
            "Global Sea Surface Temp anomalies (°C) by year, vs. 1971-2000 mean"
            if args.dataset == "anom"
            else "Global Sea Surface Temps (°C) by year"
        ),
        args.ice,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    try:
        parser = argparse.ArgumentParser(
            description="Climate data pipeline",
            formatter_class=CustomFormatter,
        )
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument(
            "--source",
            choices=sorted(SOURCES.keys()),
            default="oisst",
            help="Data source",
        )
        parser.add_argument(
            "--dataset", "-d",
            default="anom",
            help="Dataset id within the source (e.g. sst, anom, t2m). "
                 "Valid values depend on --source.",
        )
        parser.add_argument(
            "--mode", "-m",
            choices=("graph", "map", "texture"),
            default="graph",
        )
        parser.add_argument(
            "--show", "-s",
            choices=("default", "ice", "land", "area"),
            default="default",
            help="OISST debug overlay (only honored by sources that implement it)",
        )
        parser.add_argument("--ice", "-i", type=bool, help="Mask cells with ice>50%%")
        parser.add_argument("--year", "-Y", type=int, default=datetime.date.today().year)
        parser.add_argument("--month", "-M", type=int, default=datetime.date.today().month)
        parser.add_argument("--day", "-D", type=int, default=datetime.date.today().day)
        parser.add_argument("--days-ago", type=int, default=0)
        parser.add_argument("--out", "-o", type=pathlib.Path)
        parser.add_argument("--cache-file", type=pathlib.Path, default="./data-cache.json")
        parser.add_argument("--start-year", type=int, default=1982)
        parser.add_argument("--dpi", type=int, default=150)
        args = parser.parse_args(argv)

        global dpi
        dpi = args.dpi

        source = SOURCES[args.source]()
        if args.dataset not in source.datasets:
            print(
                f"❌ Source '{args.source}' has no dataset '{args.dataset}'. "
                f"Valid: {sorted(source.datasets.keys())}",
                file=sys.stderr,
            )
            return 1

        load_cache(args.cache_file)

        if not args.out:
            plt.ion()  # interactive mode

        if args.mode == "graph":
            asyncio.run(process_all(source, args))
        else:
            asyncio.run(process_map(source, args))

        if not args.out:
            plt.show(block=True)

    except DataFetchError as e:
        print(f"❌Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
