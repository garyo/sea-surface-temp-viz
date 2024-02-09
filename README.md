# Sea Surface Temp Visualization

This python code downloads daily worldwide sea surface temp datasets
and visualizes them either as a world map on a given day, or as a time
series per year.

It downloads the sea surface temp data from
https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr.
That contains daily HDF datasets containing .25° lat/lon cells over
the entire Earth surface for each day since 1982, including temp and
temperature anomaly compared to the historical mean for that date.

This code processes those datasets into maps for any given day, or
line charts with temp per day for the entire time period. The processing steps include:
- limiting to +/- 60° latitude, to ignore the polar regions
- removing cells containing > 50% ice cover
- weighting each cell by its actual area (cells closer to the poles are smaller)

Latest files, auto-produced daily by a github action. You can use these directly via these URLs:
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-map.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-anom-map.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all.png
- https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all-anom.png

![SST temp map](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-map.png)
![SST anomaly map](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-anom-map.png)
![SST temp graph](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all.png)
![SST anomaly graph](https://github.com/garyo/sea-surface-temp-viz/releases/latest/download/sst-all-anom.png)

For older examples, see the `doc` subdir.

