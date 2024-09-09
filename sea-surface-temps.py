# SPDX-License-Identifier: MIT

import sys
import requests
import io
import urllib3
import asyncio
import aiohttp
import datetime
import calendar
import json
import argparse
import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import DefaultDict
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

n_concurrent_requests = 20
dpi = 150


# rescale x, in the range [oldmin,oldmax], into [newmin,newmax]
def rescale(x, oldmin, oldmax, newmin, newmax):
    return ((x - oldmin) / (oldmax - oldmin)) * (newmax - newmin) + newmin


# Area of a given lat/lon square, relative to the size at the equator
# Varies as cos(latitude)
def rel_area(lat, _):
    latitude_radians = np.radians(lat)
    return np.cos(latitude_radians)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if type(o) is np.float32:
            return o.item()
        return json.JSONEncoder.default(self, o)


def get_historical_data():
    url = (
        "https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_world2_sst_day.json"
    )
    response = requests.get(url)
    if response.status_code == 200:
        historical_data = json.loads(response.content)

        # predicate = lambda x: x["name"] == "1982-2011 mean"
        def predicate(x):
            return x["name"] == "1982-2011 mean"

        mean_sst_1982_2011 = next(filter(predicate, historical_data))["data"]
        return mean_sst_1982_2011


get_historical_data()


# Return data from the first URL that succeeds
async def try_fetch(urls, session):
    for url in urls:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    print(f"Failed to fetch {url}: status={response.status}")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            # keep going
    raise ValueError("All URL fetches failed")


async def get_sst_dataset(year, mo, day, session, semaphore):
    # these are about 1.5MB
    # 'anom': difference from 1971-2000 mean for that grid cell
    url_base = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
    url_final = f"{url_base}/{year:04}{mo:02}/oisst-avhrr-v02r01.{year:04}{mo:02}{day:02}.nc"
    url_prelim = f"{url_base}/{year:04}{mo:02}/oisst-avhrr-v02r01.{year:04}{mo:02}{day:02}_preliminary.nc"

    async with semaphore:
        hdf = None
        try:
            print(f"Requesting data for {year}-{mo}-{day}...")
            data = await try_fetch([url_final, url_prelim], session)
            # Use io.BytesIO to create a file-like object from the downloaded bytes
            file_obj = io.BytesIO(data)
            hdf = h5py.File(file_obj, "r")
            print(f"Got hdf from {year}-{mo}-{day}")
            return hdf
        except ValueError:
            print(f"Failed to download {year}-{mo}-{day} from any URL.")
            raise


# Get HDF dataset, mask out where ice > 50% and array == -999, and apply scale_factor
# Returns a masked array, so make sure to use `np.ma` functions on it.
def get_processed_hdf_data_array(
    hdf, dataset_name, lat_min, lat_max, use_ice_mask=False, show="default"
):
    array = hdf[dataset_name][0][0][:][:]
    scale_factor = hdf[dataset_name].attrs["scale_factor"]
    assert hdf[dataset_name].attrs["add_offset"] == 0
    lat = hdf["lat"][:]
    lon = hdf["lon"][:]
    (lon, lat) = np.meshgrid(lon, lat)
    # note: mask True means invalid data (so "masked out")
    if use_ice_mask:
        ice = hdf["ice"][0][0][:][:]
    else:
        ice = False
    ice_mask = ice > 50
    # array == -999 indicates land rather than sea
    data_mask = array == -999
    lat_mask = np.logical_or(lat < lat_min, lat > lat_max)
    if show == "ice":
        return ice
    if show == "land":
        return array == -999
    if show == "area":
        return rel_area(lat, lon)
    else:  # default: show dataset
        masked_array = np.ma.array(
            array,
            mask=np.ma.mask_or(lat_mask, np.ma.mask_or(ice_mask, data_mask)),
        )
        return masked_array * scale_factor


def latlon_weights(hdf):
    lat = hdf["lat"][:]
    lon = hdf["lon"][:]
    (lon, lat) = np.meshgrid(lon, lat)
    return rel_area(lat, lon)


def get_average_temp(hdf, dataset_name, use_ice_mask):
    data = get_processed_hdf_data_array(hdf, dataset_name, -60, 60, use_ice_mask)
    weights = latlon_weights(hdf)
    average = np.ma.average(data, weights=weights)
    return average


# Disk cache: entries are yyyy-mm-dd-DATASETNAME
temps_cache_file = "./sst-data-cache.json"
temps_cache = {}


def save_cache():
    global temps_cache
    global temps_cache_file
    json_data = json.dumps(temps_cache, sort_keys=True, indent=2, cls=NumpyArrayEncoder)
    with open(temps_cache_file, "w") as f:
        f.write(json_data)


def load_cache(path):
    global temps_cache_file
    global temps_cache
    temps_cache_file = path
    try:
        with open(temps_cache_file, "r") as f:
            temps_cache = json.load(f)
    except IOError:
        temps_cache = {}


async def get_temp_for_date(
    year, mo, day, dataset_name, use_ice_mask, session, semaphore, lock
):
    async with lock:

        def cache_key(name):
            return f"{year}-{mo:02}-{day:02}-{name}"

        cached = temps_cache.get(cache_key(dataset_name))
        if cached:
            # print(f"Average {dataset_name} {year}-{mo}-{day}: {cached:.4f}°C (from cache)")
            return (year, mo, day, cached)

    try:
        hdf = await get_sst_dataset(year, mo, day, session, semaphore)
        # compute & cache both datasets
        t_sst = get_average_temp(hdf, "sst", use_ice_mask)
        t_anom = get_average_temp(hdf, "anom", use_ice_mask)
        temps_cache[cache_key("sst")] = t_sst
        temps_cache[cache_key("anom")] = t_anom
        t = t_sst if dataset_name == "sst" else t_anom
        print(f"Average {dataset_name} {year}-{mo}-{day}: {t:.4f}°C")
        do_save = True
    except ValueError:
        t = np.nan
        do_save = False

    if do_save:
        async with lock:
            save_cache()
    return (year, mo, day, t)


def plot_globe_dataset(data, hdf, vmin, vmax, cmap, title):
    # Set up the map projection and figure
    # 'mollweide' is good
    _, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "mollweide"})

    ax.grid(visible=True)

    lat_1d = hdf["lat"][:]  # -90 to 90
    lon_1d = hdf["lon"][:]  # 0 to 360

    # Plot the data
    # mollweide requires longitude in radians -pi to pi, latitude in radians -pi/2 to pi/2
    # So we have to rotate the data around the longitude axis, then shift longitude to match
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

    # Add a colorbar
    plt.colorbar(c, orientation="horizontal", pad=0.05)

    plt.title(title)
    plt.suptitle(
        f"Created at {datetime.datetime.now()}\nCopyright {datetime.date.today().year} Gary Oberbrunner",
        fontsize=7,
        y=0.97,
    )
    plt.tight_layout()
    return plt


def plot_equirect_dataset(data, vmin, vmax, cmap, outfile):
    plt.imsave(outfile, data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")


def rescale_colormap_def_to_01(cmap):
    # colormap is a list of pairs, each pair containing x-value and color
    # this utility remaps the x-values into 0,1
    vmin = cmap[0][0]
    vmax = cmap[-1][0]
    return [[rescale(x[0], vmin, vmax, 0, 1), *x[1:]] for x in cmap]


def save_metadata(metadata, outfile):
    metadata_json = json.dumps(metadata)
    p = pathlib.Path(outfile)
    dir = p.parent
    name = p.stem
    outfile = dir / (name + "-metadata.json")
    outfile.write_text(metadata_json)


# Create map or equirectangular texture
async def process_map(args):
    if args.days_ago:
        date = datetime.date.today() - datetime.timedelta(days=args.days_ago)
        year = date.year
        mo = date.month
        day = date.day
    else:
        year = args.year
        mo = args.month
        day = args.day
    date = f"{year}-{mo:02}-{day:02}"
    semaphore = asyncio.Semaphore(n_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        try:
            hdf = await get_sst_dataset(year, mo, day, session, semaphore)
        except ValueError:
            print(f"Failed to get SST data for {year}-{mo}-{day}")
            raise

    if args.dataset == "anom":
        data = get_processed_hdf_data_array(hdf, "anom", -90, 90, args.ice, args.show)

        # Blue below zero (midpoint=0.5), yellow to red above. Midpoint should map to 0 temp diff.
        cmapdef = [
            [-3, "darkblue"],
            [-0.5, "lightblue"],
            [0, "white"],
            [0.5, "yellow"],
            [2.5, "red"],
            [3, "darkred"],
            [4.0, "white"],
            [5.0, "gray"],
            [7.0, "violet"],
        ]
        variance_cmap = LinearSegmentedColormap.from_list(
            "sst_cmap", rescale_colormap_def_to_01(cmapdef)
        )
        domain_min = cmapdef[0][0]
        domain_max = cmapdef[-1][0]
        title = f"{date}\nSea Surface Temp Variance from 1971–2000 Mean, °C"
        if args.mode == "map":
            plot_globe_dataset(data, hdf, domain_min, domain_max, variance_cmap, title)
        else:
            # Plot an equirectangular texture that can be used on a 3d sphere,
            # and save the colormap definition and other metadata to a file
            # so that can be used in the sphere visualization
            plot_equirect_dataset(data, domain_min, domain_max, variance_cmap, args.out)
            metadata = {
                "cmap": cmapdef,
                "title": title,
                "dataset": args.dataset,
                "date": date,  # YYYY-MM-DD
                "year": year,
                "month": mo,
                "day": day,
            }
            if args.out:
                save_metadata(metadata, args.out)
    else:

        data = get_processed_hdf_data_array(hdf, "sst", -90, 90, args.ice, args.show)

        # range 0-35°C
        # white at 20°C
        cmapdef = [
            [0, "darkblue"],
            [20, "white"],
            [22.5, "yellow"],
            [30, "red"],
            [32, "darkred"],
            [35, "white"],
        ]
        sst_cmap = LinearSegmentedColormap.from_list(
            "sst_cmap", rescale_colormap_def_to_01(cmapdef)
        )
        domain_min = cmapdef[0][0]
        domain_max = cmapdef[-1][0]
        title = f"{date}\nSea Surface Temp, °C"
        if args.mode == "map":
            plot_globe_dataset(data, hdf, domain_min, domain_max, sst_cmap, title)
        else:
            plot_equirect_dataset(data, domain_min, domain_max, sst_cmap, args.out)
            metadata = {
                "cmap": cmapdef,
                "title": title,
                "dataset": args.dataset,
                "date": date,  # YYYY-MM-DD
                "year": year,
                "month": mo,
                "day": day,
            }
            if args.out:
                save_metadata(metadata, args.out)

    if args.out and args.mode == "map":
        plt.savefig(args.out, dpi=dpi)
    elif args.out and args.mode == "texture":
        pass  # already saved
    else:
        plt.show()


def create_cache_dict() -> DefaultDict[int, DefaultDict[int, DefaultDict[int, float]]]:
    """Function returning 3-level dictionary [key][key][key] -> float, all levels "defaultdict"
    so they get created on use.
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

async def process_all(args):
    async def get_data(dataset_name, session, semaphore, lock):
        start_year = args.start_year
        end_year = datetime.date.today().year
        tasks = []

        # Get all the data, into a 3-level dict: results[year][month][day] = val
        for year in range(start_year, end_year + 1):
            for mo in range(1, 13):  # 12 months in a year
                num_days = calendar.monthrange(year, mo)[1]
                for day in range(1, num_days + 1):
                    if datetime.date(year, mo, day) > datetime.date.today():
                        continue
                    try:
                        tasks.append(
                            asyncio.create_task(
                                get_temp_for_date(
                                    year,
                                    mo,
                                    day,
                                    dataset_name,
                                    args.ice,
                                    session,
                                    semaphore,
                                    lock,
                                )
                            )
                        )
                    except IOError:
                        print(f"Skipping {year}-{mo}-{day}: failed to get data.")
                        pass
        results = create_cache_dict()
        for task in asyncio.as_completed(tasks):
            year, mo, day, val = await task
            async with lock:
                results[year][mo][day] = val
        return results

    year_cmap = LinearSegmentedColormap.from_list(
        "year_cmap", ["lightgray", "darkblue"]
    )

    # Convert day of year (0-based) to date
    def year_day_to_date(year, day):
        start_date = datetime.date(year, 1, 1)
        return start_date + datetime.timedelta(days=day)  # when day=0, it's Jan 1st

    # Convert year, month, day to 0-based day number
    # ONLY USE THIS FOR GRAPHS; it aligns all days after Feb with leap years.
    def ymd_to_year_day_for_graph(year, mo, day):
        while not calendar.isleap(year):
            year += 1           # find the next leap year
        return datetime.date(year, mo, day).timetuple().tm_yday - 1

    def plot_fig(temps, title, use_ice_mask):
        _, ax = plt.subplots(figsize=(14, 8))
        years = np.array(sorted(list(temps.keys())))
        record = [-10000, (0, 0, 0)]  # value, then year, month, day

        # N years ago (or oldest), 0 means current
        def years_ago(n):
            if n >= len(years):
                return years[0]
            return years[-(n + 1)]

        for year in years:
            y = []
            x = []
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
                color = "red"
                linewidth = 2
            if year == years_ago(1):
                color = "orange"
                linewidth = 1.5
            if year == years_ago(2):
                color = "green"
            if year > years_ago(4) or year < years[2]:
                label = f"{year}"
            else:
                label = None
            ax.plot(x, y, label=label, color=color, linewidth=linewidth)
            if year == years[-1]:
                # mark last point
                last_date = year_day_to_date(year, x[-1])
                last_color = "blue"
                ax.annotate(
                    f"{last_date}\n{y[-1]:.2f}°C",
                    xy=(x[-1], y[-1]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    verticalalignment="top",
                    horizontalalignment="left",
                    color=last_color,
                )
                ax.plot(x[-1], y[-1], marker=".", markersize=5, color=last_color)
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
            f"Years: {years[0]}-{years[-1]}. Created {datetime.datetime.now()}\nCopyright {datetime.date.today().year} Gary Oberbrunner",
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
        plt.text(
            0,
            0,
            msg,
            ha="left",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=9,
        )
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=dpi)
        else:
            plt.show()

    semaphore = asyncio.Semaphore(n_concurrent_requests)
    lock = asyncio.Lock()
    type = args.dataset  # 'sst' or 'anom' for anomaly compared to 1971-2000 baseline
    async with aiohttp.ClientSession() as session:
        temp_data_by_date = await get_data(type, session, semaphore, lock)

    plot_fig(
        temp_data_by_date,
        (
            "Global Sea Surface Temp anomalies (°C) by year, vs. 1971-2000 mean"
            if type == "anom"
            else "Global Sea Surface Temps (°C) by year"
        ),
        args.ice,
    )


def main(argv=None):
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    try:
        parser = argparse.ArgumentParser(
            description="""Sea Surface Temperature Visualizer""",
            formatter_class=CustomFormatter,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="""Process verbosely""",
        )
        parser.add_argument(
            "--dataset",
            "-d",
            choices=("anom", "sst"),
            default="anom",
            help="""Dataset: sst=temperatures, anom=anomalies vs. mean""",
        )
        parser.add_argument(
            "--mode",
            "-m",
            choices=("graph", "map", "texture"),
            default="graph",
            help="""Mode: graph=graph of all time, map=map of today, texture=equirect texture for globe""",
        )
        parser.add_argument(
            "--show",
            "-s",
            choices=("default", "ice", "land", "area"),
            default="default",
            help="""Show: default=temps, ice=ice mask, land=land mask (only in map mode), area=rel cell area""",
        )
        parser.add_argument(
            "--ice", "-i", type=bool, help="""Mask cells with ice>50%%"""
        )
        parser.add_argument(
            "--year",
            "-Y",
            type=int,
            default=datetime.date.today().year,
            help="""Year for map mode""",
        )
        parser.add_argument(
            "--month",
            "-M",
            type=int,
            default=datetime.date.today().month,
            help="""Month for map mode""",
        )
        parser.add_argument(
            "--day",
            "-D",
            type=int,
            default=datetime.date.today().day,
            help="""Day of month for map mode""",
        )
        parser.add_argument(
            "--days-ago",
            type=int,
            default=0,
            help="""Days ago (before today), for map mode""",
        )
        parser.add_argument(
            "--out",
            "-o",
            type=pathlib.Path,
            help="""Output image to this path""",
        )
        parser.add_argument(
            "--cache-file",
            type=pathlib.Path,
            default="./sst-data-cache.json",
            help="""Cache file to speed up future runs""",
        )
        parser.add_argument(
            "--start-year",
            type=int,
            default=1982,
            help="""Start year for map mode""",
        )
        parser.add_argument(
            "--dpi",
            type=int,
            default=150,
            help="""DPI for output figures (sets resolution)""",
        )
        args = parser.parse_args(argv)

        global dpi
        dpi = args.dpi

        load_cache(args.cache_file)

        if not args.out:
            plt.ion  # interactive mode

        if args.mode == "graph":
            asyncio.run(process_all(args))
        else:  # map or texture
            asyncio.run(process_map(args))

        if not args.out:
            plt.show(block=True)  # run event loop til all windows closed

    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
