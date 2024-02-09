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

n_concurrent_requests = 10

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
            data = await try_fetch([url_prelim, url_final], session)
            # Use io.BytesIO to create a file-like object from the downloaded bytes
            file_obj = io.BytesIO(data)
            hdf = h5py.File(file_obj, 'r')
            print(f'Got hdf from {year}-{mo}-{day}')
            return hdf
        except ValueError:
            print(f"Failed to download {year}-{mo}0-{day}.")
            raise

# Get HDF dataset, mask out where ice > 50% and array == -999, and apply scale_factor
# Note: does not weight samples by area.
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
    verbose = False
    if verbose:
        print(f'lat_mask({lat_min},{lat_max}): {lat_mask[360:365,720:725]}')
    masked_array = np.ma.array(array, mask=np.ma.mask_or(lat_mask, np.ma.mask_or(ice_mask, data_mask)))
    return masked_array * scale_factor

# Gets the dataset from the given HDF
# Returned dataset is properly masked and weighted
def get_average_temp(hdf, dataset_name):
    lat = hdf['lat'][:]
    lon = hdf['lon'][:]
    (lon, lat) = np.meshgrid(lon, lat)

    data = get_processed_hdf_data_array(hdf, dataset_name, -60, 60)
    weighted = data * rel_area(lat, lon) # this is now 1d
    average = np.average(weighted)
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
        cache_key = f'{year}-{mo:02}-{day:02}-{dataset_name}'
        cached = temps_cache.get(cache_key)
        if cached:
            # print(f"Average {dataset_name} {year}-{mo}-{day}: {cached:.4f}°C (from cache)")
            return (year, mo, day, cached)

    try:
        hdf = await get_sst_dataset(year, mo, day, session, semaphore)
        t = get_average_temp(hdf, dataset_name)
        do_save = True
    except ValueError:
        t = np.nan
        do_save = False

    if do_save:
        async with lock:
            temps_cache[cache_key] = t
            save_cache()
    print(f"Average {dataset_name} {year}-{mo}-{day}: {t:.4f}°C")
    return (year, mo, day, t)

def plot_globe_dataset(data, hdf, vmin, vmax, cmap, title):
    # Set up the map projection and figure
    # 'mollweide' is good
    fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'})

    # Set extent (optional, to zoom into a specific area)
    # ax.set_extent([-180, 180, -60, 60], crs=proj)

    # Add coastlines for reference
    # ax.coastlines()

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
    return plt

def process_date(args):
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
    try:
        hdf = get_sst_dataset(year, mo, day, session, semaphore)
    except IOError:
        print(f"Failed to get SST data for {year}-{mo}-{day}")
        raise

    if args.dataset == 'anom':
        data = get_processed_hdf_data_array(hdf, 'anom', -90, 90)
        # Blue below zero (midpoint=0.5), yellow to red above. Midpoint should map to 0 temp diff.
        variance_cmap = LinearSegmentedColormap.from_list("sst_cmap",
                                                          [[0, "white"], [0.2, "darkblue"],
                                                           [0.45, "lightblue"], [0.5, "white"],
                                                           [0.6, "yellow"], [0.9, "red"],
                                                           [1.0, "darkred"]])
        plot_globe_dataset(data, hdf, -3, 3, variance_cmap, f'{date}\nSea Surface Temp Variance from 1971–2000 Mean, °C')
    else:
        data = get_processed_hdf_data_array(hdf, 'sst', -90, 90)
        # white at 20°C or 0.666
        sst_cmap = LinearSegmentedColormap.from_list("sst_cmap",
                                                     [[0, "darkblue"], [0.666, "white"],
                                                      [0.8, "yellow"], [0.9, "red"],
                                                      [1.0, "darkred"]])
        plot_globe_dataset(data, hdf, 0, 30, sst_cmap, f'{date}\nSea Surface Temp, °C')
    if args.out:
        plt.savefig(args.out, dpi=300)
    else:
        plt.show()


# Define a function to create 3-level dictionary of ints
def create_multilevel_dict_with_ints():
    from collections import defaultdict

    def nested_dict(n, default_type=int):
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

        # Get all the data
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

    def plot_fig(temps, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        years = np.array(sorted(list(temps.keys())))
        for year in years:
            data = temps[year]
            data = [x for x in data if not np.isnan(x)]

            cmap_index = (year - years[0]) / (years[-1] - years[0])
            linewidth = 1
            color = year_cmap(cmap_index)
            if year == years[-1]:
                color = "red"
                linewidth = 2
            if year == years[-2]:
                color = "orange"
                linewidth = 2
            if year == years[-3]:
                color = "green"
            if year > years[-5] or year < years[2]:
                label = f'{year}'
            else:
                label = None
            ax.plot(data, label=label, color=color, linewidth=linewidth)
            if year == years[-1]:
                # mark last point
                plt.text(len(data), data[-1], f'{data[-1]:.2f}',
                         verticalalignment='bottom', horizontalalignment='left')
                plt.plot(len(data), data[-1], marker='.', markersize=5, color='black')
        plt.title(title)
        plt.suptitle(f'Years: {years[0]}-{years[-1]}. Created {datetime.datetime.now()}', fontsize=5, y=0.97)
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
        if args.out:
            plt.savefig(args.out, dpi=300)
        else:
            plt.show()

    semaphore = asyncio.Semaphore(n_concurrent_requests)
    lock = asyncio.Lock()
    type = args.dataset # 'sst' or 'anom' for anomaly compared to 1971-2000 baseline
    async with aiohttp.ClientSession() as session:
        temp_data_by_date = await get_data(type, session, semaphore, lock)

    # postprocess into what the plot needs (by year, then day of year)
    temp_data = {}
    for year in sorted(list(temp_data_by_date.keys())):
        temp_data[year] = []
        for month in sorted(list(temp_data_by_date[year].keys())):
            for day in sorted(list(temp_data_by_date[year][month].keys())):
                temp_data[year].append(temp_data_by_date[year][month][day])

    plot_fig(temp_data,
             "Sea Surface Temp anomalies (°C) by year, vs. 1971-2000 mean" if type == 'anom'
             else "Sea Surface Temps by year")


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
                            default = 2000,
                            help="""Start year for map mode""")
        args = parser.parse_args(argv)

        load_cache(args.cache_file)
        if args.mode == 'all':
            asyncio.run(process_all(args))
        else:
            process_date(args)
    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
