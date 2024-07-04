# Sea Surface Temperature Visualization

Sea surface temperatures are a key indicator of climate change. The
world's oceans are a significant reservoir of heat, warming and
cooling much more slowly than the atmosphere. Ocean heat drives larger
storms, adds more moisture to the air above the ocean, and affects sea
life and ocean chemistry. Monitoring ocean temperature is important to
understand the drivers and impacts of climate change.

NOAA's [National Centers for Environmental
Information](https://www.ncei.noaa.gov/) maintains datasets of daily
global ocean temperatures, derived mostly from satellites and updated
using ship and buoy based observations. The raw data is filtered and
resampled onto a 1/4-degree grid over the whole globe, and is
available for every day from 1981 through the present at [their
website](https://www.ncei.noaa.gov/products/climate-data-records/sea-surface-temperature-optimum-interpolation)
with one or two days lag. It is an amazing testament to the incredible
work of scientists and engineers worldwide that we have access to
real-time data like this.

This tool produces a set of visualizations of that data: maps of the
latest temperature, and a set of time series of average temps for all
years. For each of those, it shows the actual temperatures as well as
the variance or "anomaly" from the mean value from the 1971-2000 time
period. The anomalies are computed cell by cell by subtracting the
mean value at that grid cell over the 1971-2000 period.

![SST temp graph](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all.png)
![SST anomaly graph](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all-anom.png)
![SST temp map](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-map.png)
![SST anomaly map](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-anom-map.png)

To see the above maps on a 3d interactive globe, visit https://globe-viz.oberbrunner.com.

Several things stand out from the time series graphs:

- Ocean temperatures have been steadily increasing over the 40-year
  recorded time span, visible in the graph as the later years are
  shaded darker blue.
- Sea surface temperatures have been rising significantly since 2023,
  reaching daily records every day since spring of that year. (This
  acceleration beyond the global climate change signal is largely due
  to [El Niño](https://oceanservice.noaa.gov/facts/ninonina.html), but
  also somewhat to the underwater eruption of the [Hunga-Tonga
  volcano](https://volcano.si.edu/volcano.cfm?vn=243040) in 2022 and
  slightly affected by the 2020 [regulation of
  sulfur](https://www.imo.org/en/MediaCentre/PressBriefings/pages/02-IMO-2020.aspx)
  in marine fuels, which had been masking temperature rises.)
- The all-time record for temperature was set in April 2024, and the
  all-time record for temperature variance from the mean was set in
  January 2024. Both records are likely to be broken as 2024
  progresses.

For technical details on data sources and how this code works, see [below](#technical-details).

**NOTE**: As of July 4, 2024, the time-series graphs are now
date-aligned, so July 4 2024 lines up with July 4 2023, even though
2024 is a leap year. This is done by "stretching" the non-leap year
dates, which avoids skipping any data. Previously, the graph's X axis
was just the day number in the year.

**NOTE**: As of April 27, 2024, the time-series graphs do _not_ mask
out cells with ice > 50%. After some discussion with the folks at
[Climate Reanalyzer](https://climatereanalyzer.com), we noticed that
on some dates, the original dataset from NOAA is missing the ice map
(it's false everywhere). And further, masking out ice cells did not
seem scientifically reasonable. This change does significantly affect
record temp dates. You can see all the historical graphs on this site,
to compare the results before and after this change. The python code
can optionally still use ice masking if you pass the `-i` command-line
argument.

Here are the latest files, auto-produced daily by a github action (run
daily at 15:05 UTC, just after the daily data update from NOAA). You can use
these directly on the web via these URLs:

- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-map.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-anom-map.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all-anom.png


For older examples, see the `doc` subdir.

# Technical Details

This python code downloads daily worldwide sea surface temp datasets
and visualizes them either as a world map on a given day, or as a time
series per year.

It downloads the sea surface temp data from
https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr.
That contains daily HDF datasets containing .25° lat/lon cells over
the entire Earth surface for each day since 1982, including temp and
temperature anomaly compared to the historical mean for that date.

It processes those datasets into maps for any given day, or
line charts with temp per day for the entire time period. The processing steps include:
- limiting to +/- 60° latitude, to ignore the polar regions
- (optionally) removing cells containing > 50% ice cover (off by default since 2024-04-27)
- weighting each cell by its actual area (cells closer to the poles are smaller)

Because it takes a while to download all the datasets, it keeps a disk
cache in `./sst-data-cache.json`. If you want to re-run from scratch,
just delete that file. It does process datasets in parallel using
`asyncio` but to get all the global datasets since 1981 takes around
20 minutes on my machine.

You can also use the included [prune-cache.py](./prune-cache.py)
utility to prune out the most recent N days from the cache so those
will be re-fetched on the next run. This is useful because sometimes
the dataset's recent data is re-analyzed to fix data quality issues.
The automatic updates of the images above are set to always re-fetch
the most recent 90 days of data.
