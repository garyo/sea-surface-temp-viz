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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

n_concurrent_requests = 15
dpi=150

# rescale x, in the range [oldmin,oldmax], into [newmin,newmax]
def rescale(x, oldmin, oldmax, newmin, newmax):
    return ((x-oldmin)/(oldmax-oldmin)) * (newmax-newmin) + newmin

# Area of a given lat/lon square, relative to the size at the equator
def rel_area(lat, lon):
    import math

    # Constants
    degree_to_radians = math.pi / 180  # Conversion factor from degrees to radians
    latitude_radians = lat * degree_to_radians
    square_area_at_equator = 1 / 360 / 360 # 1 degree square of unit sphere
    # Width per degree of latitude (constant)
    width_per_degree_latitude = 1 / 360
    # Width per degree of longitude (varies with latitude)
    width_per_degree_longitude = np.cos(latitude_radians) * width_per_degree_latitude
    # Area of a square degree at the given latitude
    return width_per_degree_latitude * width_per_degree_longitude / square_area_at_equator

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

def get_historical_data():
    url = 'https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_world2_sst_day.json'
    response = requests.get(url)
    if response.status_code == 200:
        historical_data = json.loads(response.content)
        predicate = lambda x: x['name'] == '1982-2011 mean'
        mean_sst_1982_2011 = next(filter(predicate, historical_data))['data']
        # print(mean_sst_1982_2011)
get_historical_data()

# Return data from the first URL that succeeds
async def try_fetch(urls, session):
    for url in urls:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            # keep going
    raise ValueError("All URL fetches failed")


async def get_sst_dataset(year, mo, day, session, semaphore):
    # these are about 1.5MB
    # 'anom': difference from 1971-2000 mean for that grid cell
    url_final = f'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year:04}{mo:02}/oisst-avhrr-v02r01.{year:04}{mo:02}{day:02}.nc'
    url_prelim = f'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year:04}{mo:02}/oisst-avhrr-v02r01.{year:04}{mo:02}{day:02}_preliminary.nc'

    async with semaphore:
        hdf = None
        try:
            print(f'Requesting data for {year}-{mo}-{day}...')
            data = await try_fetch([url_final, url_prelim], session)
            # Use io.BytesIO to create a file-like object from the downloaded bytes
            file_obj = io.BytesIO(data)
            hdf = h5py.File(file_obj, 'r')
            print(f'Got hdf from {year}-{mo}-{day}')
            return hdf
        except ValueError:
            print(f"Failed to download {year}-{mo}0-{day}.")
            raise

# Get HDF dataset, mask out where ice > 50% and array == -999, and apply scale_factor
# Returns a masked array, so make sure to use `np.ma` functions on it.
def get_processed_hdf_data_array(hdf, dataset_name, lat_min, lat_max):
    array = hdf[dataset_name][0][0][:][:]
    scale_factor = hdf[dataset_name].attrs["scale_factor"]
    assert(hdf[dataset_name].attrs['add_offset'] == 0)
    lat = hdf['lat'][:]
    lon = hdf['lon'][:]
    (lon, lat) = np.meshgrid(lon, lat)
    ice = hdf['ice'][0][0][:][:]
    # note: mask True means invalid data (so "masked out")
    ice_mask = ice > 50
    data_mask = array == -999
    lat_mask = np.logical_or(lat < lat_min, lat > lat_max)
    masked_array = np.ma.array(array, mask=np.ma.mask_or(lat_mask, np.ma.mask_or(ice_mask, data_mask)))
    return masked_array * scale_factor

def latlon_weights(hdf):
    lat = hdf['lat'][:]
    lon = hdf['lon'][:]
    (lon, lat) = np.meshgrid(lon, lat)
    return rel_area(lat, lon)

def get_average_temp(hdf, dataset_name):
    data = get_processed_hdf_data_array(hdf, dataset_name, -60, 60)
    weights = latlon_weights(hdf)
    average = np.ma.average(data, weights=weights)
    return average

# Disk cache: entries are yyyy-mm-dd-DATASETNAME
temps_cache_file='./sst-data-cache.json'
temps_cache = {}

def save_cache():
    global temps_cache
    global temps_cache_file
    json_data = json.dumps(temps_cache, cls=NumpyArrayEncoder)
    with open(temps_cache_file, 'w') as f:
        f.write(json_data)

def load_cache(path):
    global temps_cache_file
    global temps_cache
    temps_cache_file = path
    try:
        with open(temps_cache_file, 'r') as f:
            temps_cache = json.load(f)
    except IOError:
        temps_cache = {}

async def get_temp_for_date(year, mo, day, dataset_name, session, semaphore, lock):
    async with lock:
        def cache_key(name):
            return f'{year}-{mo:02}-{day:02}-{name}'
        cached = temps_cache.get(cache_key(dataset_name))
        if cached:
            # print(f"Average {dataset_name} {year}-{mo}-{day}: {cached:.4f}°C (from cache)")
            return (year, mo, day, cached)

    try:
        hdf = await get_sst_dataset(year, mo, day, session, semaphore)
        # compute & cache both datasets
        t_sst = get_average_temp(hdf, 'sst')
        t_anom = get_average_temp(hdf, 'anom')
        temps_cache[cache_key('sst')] = t_sst
        temps_cache[cache_key('anom')] = t_anom
        t = t_sst if dataset_name == 'sst' else t_anom
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
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': 'mollweide'})

    ax.grid(visible=True)

    lat_1d = hdf['lat'][:]      # -90 to 90
    lon_1d = hdf['lon'][:]      # 0 to 360

    # Plot the data
    # mollweide requires longitude in radians -pi to pi, latitude in radians -pi/2 to pi/2
    # So we have to rotate the data around the longitude axis, then shift longitude to match
    data_for_map = np.roll(data, data.shape[1]//2, axis=1)
    lon_for_map = lon_1d - 180
    c = ax.pcolormesh(np.radians(lon_for_map), np.radians(lat_1d), data_for_map,
                      cmap=cmap,
                      vmin=vmin, vmax=vmax)

    # Add a colorbar
    plt.colorbar(c, orientation='horizontal', pad=0.05)

    plt.title(title)
    plt.suptitle(f'Created at {datetime.datetime.now()}\nCopyright {datetime.date.today().year} Gary Oberbrunner',
                 fontsize=7, y=0.97)
    plt.tight_layout()
    return plt

def rescale_colormap_def_to_01(cmap):
    # colormap is a list of pairs, each pair containing x-value and color
    # this utility remaps the x-values into 0,1
    vmin = cmap[0][0]
    vmax = cmap[-1][0]
    return [[rescale(x[0], vmin, vmax, 0, 1), *x[1:]] for x in cmap]


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
    date=f'{year}-{mo:02}-{day:02}'
    semaphore = asyncio.Semaphore(n_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        try:
            hdf = await get_sst_dataset(year, mo, day, session, semaphore)
        except ValueError:
            print(f"Failed to get SST data for {year}-{mo}-{day}")
            raise

    if args.dataset == 'anom':
        data = get_processed_hdf_data_array(hdf, 'anom', -90, 90)

        # Blue below zero (midpoint=0.5), yellow to red above. Midpoint should map to 0 temp diff.
        cmapdef = [[-3, "darkblue"],
                   [-0.5, "lightblue"],
                   [0, "white"],
                   [0.5, "yellow"],
                   [2.5, "red"],
                   [3, "darkred"],
                   [3.5, "white"]]
        variance_cmap = LinearSegmentedColormap.from_list("sst_cmap",
                                                          rescale_colormap_def_to_01(cmapdef))
        plot_globe_dataset(data, hdf, -3, 3.5, variance_cmap, f'{date}\nSea Surface Temp Variance from 1971–2000 Mean, °C')
    else:
        data = get_processed_hdf_data_array(hdf, 'sst', -90, 90)

        # range 0-35°C
        # white at 20°C
        cmapdef = [[0, "darkblue"],
                   [20, "white"],
                   [22.5, "yellow"],
                   [30, "red"],
                   [32, "darkred"],
                   [35, "white"]]
        sst_cmap = LinearSegmentedColormap.from_list("sst_cmap",
                                                     rescale_colormap_def_to_01(cmapdef))
        plot_globe_dataset(data, hdf, 0, 35, sst_cmap, f'{date}\nSea Surface Temp, °C')
    if args.out:
        plt.savefig(args.out, dpi=dpi)
    else:
        plt.show()


# Define a function to create 3-level dictionary of ints
def create_multilevel_dict_with_ints():
    from collections import defaultdict

    def nested_dict(n, default_type=float):
        if n == 1:
            return defaultdict(default_type)
        else:
            return defaultdict(lambda: nested_dict(n-1, default_type))

    data = nested_dict(3)
    return data


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
                        tasks.append(asyncio.create_task(get_temp_for_date(year, mo, day, dataset_name, session, semaphore, lock)))
                    except IOError:
                        print(f"Skipping {year}-{mo}-{day}: failed to get data.")
                        pass
        results = create_multilevel_dict_with_ints()
        for task in asyncio.as_completed(tasks):
            year, mo, day, val = await task
            async with lock:
                results[year][mo][day] = val
        return results

    year_cmap = LinearSegmentedColormap.from_list("year_cmap", ["lightgray", "darkblue"])

    # Convert day of year (0-based) to date
    def year_day_to_date(year, day):
        start_date = datetime.date(year, 1, 1)
        return start_date + datetime.timedelta(days=day) # when day=0, it's Jan 1st

    # Convert year, month, day to 0-based day number
    def ymd_to_year_day(year, mo, day):
        return datetime.date(year, mo, day).timetuple().tm_yday - 1

    def plot_fig(temps, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        years = np.array(sorted(list(temps.keys())))
        record = [-10000, (0, 0, 0)] # value, then year, month, day
        # N years ago (or oldest), 0 means current
        def years_ago(n):
            if n >= len(years):
                return years[0]
            return years[-(n+1)]
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
                        x.append(ymd_to_year_day(year, month, day))
                        y.append(val)

            cmap_index = (year - years[0]) / (years[-1] - years[0])
            linewidth = 0.5
            color = year_cmap(cmap_index)
            if year == years_ago(0):
                color = "red"
                linewidth = 2
            if year == years_ago(1):
                color = "orange"
                linewidth = 2
            if year == years_ago(2):
                color = "green"
            if year > years_ago(4) or year < years[2]:
                label = f'{year}'
            else:
                label = None
            ax.plot(x, y, label=label, color=color, linewidth=linewidth)
            if year == years[-1]:
                # mark last point
                last_date = year_day_to_date(year, x[-1])
                last_color = "blue"
                ax.annotate(f'{last_date}\n{y[-1]:.2f}°C',
                             xy=(x[-1], y[-1]),
                             xytext=(3, 0), textcoords='offset points',
                             verticalalignment='top', horizontalalignment='left',
                             color=last_color)
                ax.plot(x[-1], y[-1], marker='.', markersize=5, color=last_color)
        if record[0] > -10000:
            record_val = record[0]
            record_x = ymd_to_year_day(*record[1])
            ax.axhline(y=record_val, color="gray", linewidth=0.5, linestyle="dashed")
            ax.plot(record_x, record_val, marker='.',
                    markersize=5, color='black')
            ax.annotate(f'record: {datetime.date(*record[1])}={record_val:.3f}',
                        xy=(record_x, record_val),
                        xytext=(0, 2), textcoords='offset points',
                        fontsize=6)
        plt.title(title)
        plt.suptitle(f'Years: {years[0]}-{years[-1]}. Created {datetime.datetime.now()}\nCopyright {datetime.date.today().year} Gary Oberbrunner',
                     fontsize=7, y=0.97)
        plt.legend(loc='lower right')
        plt.xticks([])
        plt.grid(axis='y')
        msg = """
        Data from www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr
        All samples weighted by grid size, 60°N to 60°S, ignoring ice > 50%.
        See https://www.ncei.noaa.gov/products/climate-data-records/sea-surface-temperature-optimum-interpolation
        """
        plt.text(0, 0,
                 msg,
                 ha="left", va="top", transform=plt.gca().transAxes, fontsize=9)
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=dpi)
        else:
            plt.show()

    semaphore = asyncio.Semaphore(n_concurrent_requests)
    lock = asyncio.Lock()
    type = args.dataset # 'sst' or 'anom' for anomaly compared to 1971-2000 baseline
    async with aiohttp.ClientSession() as session:
        temp_data_by_date = await get_data(type, session, semaphore, lock)

    plot_fig(temp_data_by_date,
             "Global Sea Surface Temp anomalies (°C) by year, vs. 1971-2000 mean" if type == 'anom'
             else "Global Sea Surface Temps (°C) by year")


def main(argv=None):
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass

    try:
        parser = argparse.ArgumentParser(description="""Sea Surface Temperature Visualizer""",
                                         formatter_class=CustomFormatter)
        parser.add_argument('--verbose', '-v', action='store_true',
                            help="""Process verbosely""")
        parser.add_argument('--dataset', '-d', choices=('anom', 'sst'),
                            default='anom',
                            help="""Dataset: sst=temperatures, anom=anomalies vs. mean""")
        parser.add_argument('--mode', '-m', choices=('all', 'map'),
                            default='all',
                            help="""Mode: all=all time, map=map of today""")
        parser.add_argument('--year', '-Y', type=int,
                            default=datetime.date.today().year,
                            help="""Year for map mode""")
        parser.add_argument('--month', '-M', type=int,
                            default=datetime.date.today().month,
                            help="""Month for map mode""")
        parser.add_argument('--day', '-D', type=int,
                            default=datetime.date.today().day,
                            help="""Day of month for map mode""")
        parser.add_argument('--days-ago', type=int,
                            default=0,
                            help="""Days ago (before today), for map mode""")
        parser.add_argument('--out', '-o', type=pathlib.Path,
                            help="""Output image to this path""")
        parser.add_argument('--cache-file', type=pathlib.Path,
                            default='./sst-data-cache.json',
                            help="""Cache file to speed up future runs""")
        parser.add_argument('--start-year', type=int,
                            default = 1982,
                            help="""Start year for map mode""")
        parser.add_argument('--dpi', type=int,
                            default = 150,
                            help="""DPI for output figures (sets resolution)""")
        args = parser.parse_args(argv)

        global dpi
        dpi = args.dpi

        load_cache(args.cache_file)

        if not args.out:
            plt.ion          # interactive mode

        if args.mode == 'all':
            asyncio.run(process_all(args))
        else:
            asyncio.run(process_map(args))

        if not args.out:
            plt.show(block=True) # run event loop til all windows closed

    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
