# Anomaly baselines: sources and method

The site ([globe-viz.oberbrunner.com](https://globe-viz.oberbrunner.com)) shows
temperature anomalies, meaning differences from a long-term average, against three
reference periods:

| Baseline | What it is | Where its offset comes from |
|---|---|---|
| **1971–2000** | The reference every published anomaly here is computed against | (none needed) |
| **1991–2020** | The current WMO climate normal | Each source's own record |
| **Pre-industrial (1850–1900)** | The IPCC reference for warming targets such as 1.5 °C | Long-record gridded reconstructions |

This document explains exactly how the 1991–2020 and pre-industrial anomalies are
derived, which datasets they use, how they were checked, and where they are
uncertain. The code is [`baselines.py`](../baselines.py) and
[`scripts/build_baseline_deltas.py`](../scripts/build_baseline_deltas.py).

## 1. Changing the baseline means adding an offset

Write $T(x, t)$ for the temperature at grid cell $x$ on day $t$, and $C_P(x, m)$ for
its mean over the years in period $P$ for calendar month $m$. The anomaly against
1971–2000, the one every source here publishes, is

$$A_{1971}(x,t) = T(x,t) - C_{1971\text{–}2000}(x, t)$$

The anomaly against any other period $B$ is the same thing plus an offset that
depends on place and time of year but not on the year:

$$A_B(x,t) = T(x,t) - C_B(x,t) = A_{1971}(x,t) + \Delta_B(x,t), \qquad
\Delta_B = C_{1971\text{–}2000} - C_B$$

So no climatologies need recomputing at daily resolution. We need only $\Delta_B$
as a function of place and time of year. §3.4 reduces it to five numbers per grid
cell. A region's average is a linear operation, so the same identity holds for
regional means:

$$\overline{A_B}^{\,R}(t) = \overline{A_{1971}}^{\,R}(t) + \overline{\Delta_B}^{\,R}(t)$$

The graphs therefore need just five numbers per region, series and baseline,
which the pipeline publishes alongside each time series. The globe uses the five
per-cell coefficient fields and adds $\Delta_B$ to every pixel (see §6).

## 2. 1991–2020

Every source here covers 1991–2020 completely, so each one's offset comes from its
own data.

- **ERA5** (sea-surface temperature and 2 m air temperature) and **GFS**: GFS
  anomalies are measured against an ERA5 climatology, so both use the same ERA5
  offset. It comes from ERA5 monthly means
  ([`reanalysis-era5-single-levels-monthly-means`](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means))
  for 1971–2020:

  $$\Delta_{1991}(x,m) = \frac{1}{30}\sum_{y=1971}^{2000} T(x,y,m) - \frac{1}{30}\sum_{y=1991}^{2020} T(x,y,m)$$

  The ERA5 grid is interpolated bilinearly to the site's 0.25° grid, the same way
  as the daily ERA5 maps.

- **OISST**: the offset uses NOAA's own daily OISST v2.1 climatologies (long-term
  means) for 1971–2000, which is what OISST's published anomaly subtracts, and for
  1991–2020. Both are published by NOAA PSL:
  [`sst.day.mean.ltm.1971-2000.nc`](https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1971-2000.nc) and
  [`sst.day.mean.ltm.1991-2020.nc`](https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1991-2020.nc).
  The daily difference is averaged within each calendar month. This makes our
  OISST 1991–2020 anomalies consistent with other sites that show OISST against
  1991–2020, such as [Climate Reanalyzer](https://climatereanalyzer.org/clim/sst_daily/).

**Check.** By construction, a series' mean 1971–2000 anomaly over 1991–2020 must
equal $-\overline{\Delta_{1991}}$. For the 60°S–60°N average:

| Series | Mean 1991–2020 anomaly vs 1971–2000 | $\overline{\Delta_{1991}}$ |
|---|---|---|
| OISST SST | +0.087 °C | −0.087 °C |
| ERA5 SST | +0.235 °C | −0.235 °C |
| ERA5 2 m air temp | +0.348 °C | −0.348 °C |

## 3. Pre-industrial (1850–1900)

No satellite record or reanalysis reaches back to 1850. The offset is therefore built
from the long-record, gridded reconstructions that the IPCC and Copernicus use. It
is done in two steps.

### 3.1 Anchoring at 1991–2020

Each source reaches 1991–2020 through its own offset (§2). The reconstructions then
supply the step from 1991–2020 back to 1850–1900:

$$\Delta_{\text{PI}}^{s}(x,m) = \Delta_{1991}^{s}(x,m) + R(x,m), \qquad
R = C^{\text{recon}}_{1991\text{–}2020} - C^{\text{recon}}_{1850\text{–}1900}$$

so that $A_{\text{PI}} = A_{1991} + R$. Copernicus relates ERA5 to 1850–1900 the
same way, from 1991–2020
([C3S methods](https://climate.copernicus.eu/climate-bulletin-about-data-and-analysis)).

The anchor matters most for OISST. Its "1971–2000" climatology predates OISST,
which begins in late 1981. NOAA built it from weekly OISST v2 data for 1982–2000
blended with an in-situ climatology for 1971–2000. Measured against that
climatology, OISST's mean anomaly over 1991–2020 is only +0.09 °C, against +0.24 °C
for ERA5 SST over the same years (§2). Going through 1971–2000 would carry that
difference into the pre-industrial numbers. Going through 1991–2020, a period
every dataset covers well, avoids it.

### 3.2 Reconstructions used

| Field | Dataset | Version used | Grid | Gaps filled | Licence |
|---|---|---|---|---|---|
| 2 m air temp | [Berkeley Earth Land/Ocean](https://berkeleyearth.org/data/), air temperature over sea ice | Jan 2025 file `Land_and_Ocean_LatLong1.nc` | 1° | Where data constrain the field | CC BY-NC 4.0 |
| 2 m air temp | [HadCRUT5](https://www.metoffice.gov.uk/hadobs/hadcrut5/) analysis, ensemble mean | 5.2.0.0 | 5° | Statistically, with masking | Open Government Licence v3 |
| 2 m air temp | [NOAAGlobalTemp](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp) | 6.1.0 (newest monthly file at build time) | 5° | Yes (neural network over land, ERSSTv6 over ocean) | US Government |
| SST | [ERSSTv6](https://www.ncei.noaa.gov/products/extended-reconstructed-sst), via [NOAA PSL](https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc) | v6 | 2° | Yes | No restrictions |
| SST | [COBE-SST2](https://psl.noaa.gov/data/gridded/data.cobe2.html), via NOAA PSL | as distributed by PSL | 1° | Yes (EOF reconstruction) | Acknowledge NOAA PSL and JMA |
| SST (check only) | [HadSST4](https://www.metoffice.gov.uk/hadobs/hadsst4/) median | 4.2.0.0 | 5° | No | Open Government Licence v3 |

The air-temperature trio is the one Copernicus averages for its global
pre-industrial offset. All three are *blended* records: they use sea-surface
temperature over open ocean. IPCC AR6 (WGI Cross-Chapter Box 2.3) assesses that
long-term changes in blended surface temperature and in true near-surface air
temperature differ by at most 10%. That makes the blended step a reasonable
estimate for ERA5's 2 m air temperature over the ocean.

HadSST4 has no infilling, and too few cells have data through 1850–1900 to use it
in the average. It is computed and printed as a cross-check only.

### 3.3 Processing

For each reconstruction $k$ separately:

1. **Monthly means with a coverage rule.** $C^k_P(x,m)$ is the mean over period
   $P$'s years for calendar month $m$. It is defined only where at least 70% of
   those years have data at that cell. Otherwise it is missing.
2. **Difference.** $R_k(x,m) = C^k_{1991\text{–}2020}(x,m) - C^k_{1850\text{–}1900}(x,m)$.
   All the inputs are anomalies or absolute temperatures. Either way, each
   dataset's own reference period cancels in this difference.
3. **Regrid.** $R_k$ is interpolated bilinearly, wrapping in longitude, to a
   common 1° grid.

Then, across datasets:

4. **Fill each dataset's gaps from the most complete one.** NOAAGlobalTemp is the
   most complete for air temperature. For SST, ERSSTv6 and COBE-SST2 are already
   effectively complete. Where $R_k$ is missing, it takes the most complete
   dataset's value plus a per-month bias:

   $$R_k(x,m) \leftarrow R_{\text{full}}(x,m) + b_k(m), \qquad
   b_k(m) = \frac{\sum_{x \in O} \cos\varphi_x\,[R_k - R_{\text{full}}](x,m)}{\sum_{x \in O}\cos\varphi_x}$$

   Here $O$ is the set of cells both datasets cover and $\varphi$ is latitude.
   Without this step, poorly observed regions would carry only the complete
   dataset's (lower) values, and the field would jump wherever a dataset's
   coverage ends.
5. **Average.** $R = \frac{1}{K}\sum_k R_k$, with equal weights.
6. **Fill remaining holes from the latitude band.** Any cell still missing (2.1% of
   cells for air temperature, none for SST) takes its 5° latitude band's mean for
   that month. A band mean keeps polar amplification, which a single global
   constant would not.
7. **Smooth spatially.** A Gaussian with σ ≈ 500 km is applied: σ = 4.5° in
   latitude and 4.5°/cos φ in longitude, wrapping. It is normalized over valid
   cells (ocean only, for SST), so coastal values aren't pulled toward zero.
   Before 1900, structure finer than this comes from the reconstructions'
   statistical models, not from observations.
8. **Interpolate** bilinearly to the site's 0.25° grid.

### 3.4 Through the year

The reconstructions are monthly, and no daily record reaches 1850–1900 globally.
For every field, including the 1991–2020 ones, the 12 monthly values at each cell
are reduced to a mean plus two annual harmonics. That is the least-squares fit
that keeps the first three Fourier terms. As a function of the fraction of the
year $\tau$:

$$\Delta(x,\tau) = a_0(x) + \sum_{j=1}^{2}\Big[a_j(x)\cos 2\pi j\tau + b_j(x)\sin 2\pi j\tau\Big]$$

The fit places month $m$'s mean at $\tau = (m - \tfrac12)/12$. Copernicus fits a
single harmonic to its global monthly offsets. A second harmonic allows a slightly
asymmetric seasonal cycle.

Everything downstream stores the five coefficients $(a_0, a_1, b_1, a_2, b_2)$,
per region for the graphs and per cell for the globe, and evaluates this formula
on each day at $\tau = (d + \tfrac12)/366$. Here $d$ is the site's leap-aligned
day of year (0 = 1 January, 59 = 29 February, 365 = 31 December). The curve is
smooth through the year, including across December–January, so daily values need
no interpolation. $a_0$ is the annual-mean offset.

## 4. Validation

| Quantity | This method | Published reference |
|---|---|---|
| Global 1991–2020 minus 1850–1900, surface air temp (whole globe, 3-dataset mean) | +0.846 °C | Copernicus: +0.88 °C (likely range 0.72–0.99) |
| Same, per dataset over the cells it covers | Berkeley Earth +0.874, HadCRUT5 +0.897, NOAAGlobalTemp +0.779 °C | |
| Global 1991–2020 minus 1850–1900, SST | ERSSTv6 +0.490, COBE-SST2 +0.543, mean +0.513 °C | HadSST4, well-sampled cells only (6%): +0.766 °C |
| Seasonal cycle of the global air-temp step | largest in Mar (+0.55), smallest in Aug (+0.40 °C) | Copernicus: largest in Mar, smallest in Jul |
| ERA5 2 m air temp, 2024, whole globe | +0.72 °C vs 1991–2020, +1.57 °C vs 1850–1900 | Copernicus: +0.72 and +1.60 °C |
| ERA5 2 m air temp, 2024, 60°S–60°N | +0.70 °C vs 1991–2020, +1.52 °C vs 1850–1900 | |
| ERA5 whole-globe daily mean, 2024-01-01 to 2026-11-16 (1,051 days) | absolute: −0.0003 °C mean difference, 0.0008 °C s.d.; vs 1991–2020: +0.001 °C mean, 0.02 °C s.d. | Copernicus Climate Pulse daily series |
| SST, 2024 | OISST +1.13, ERA5 +1.06 °C vs 1850–1900 (60°S–60°N) | IPCC AR6: ocean +0.88 °C by 2011–2020 |

Our air-temperature step is about 0.04 °C smaller than Copernicus's global offset.
There are two reasons. Copernicus uses each dataset's published global series,
which includes poorly observed polar cells that our 70% coverage rule leaves to
the other datasets. And NOAAGlobalTemp's grid runs lowest of the three. The
difference is well inside the spread between datasets. It accounts for the
0.03 °C gap in the 2024 pre-industrial anomaly, since the 1991–2020 anomalies
agree. Our "Global" region spans 60°S–60°N; air temperature also has a
"Whole globe" region (from 2024 on), which is the one to compare with
whole-planet figures. The daily Climate Pulse comparison confirms that our ERA5
processing reproduces Copernicus's global means; the 0.02 °C day-to-day scatter
in anomalies comes from our seasonal offset being two smooth harmonics where
Copernicus uses a day-by-day climatology.

Annual-mean offsets for each region, in °C, added to the 1971–2000 anomaly:

| Baseline / series | Global | Whole globe | Tropics | N. Hemi | S. Hemi | Niño 3.4 | Pacific | Atlantic | Indian | Arctic | Southern |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1991–2020 · OISST SST | −0.09 | — | −0.10 | −0.15 | −0.04 | −0.09 | −0.08 | −0.11 | −0.05 | −0.18 | +0.27 |
| 1991–2020 · ERA5 SST | −0.24 | — | −0.23 | −0.31 | −0.18 | −0.17 | −0.23 | −0.27 | −0.19 | −0.26 | +0.03 |
| 1991–2020 · ERA5/GFS air | −0.35 | −0.40 | −0.32 | −0.46 | −0.24 | −0.25 | −0.25 | −0.31 | −0.22 | −1.47 | −0.16 |
| Pre-industrial · OISST SST | +0.46 | — | +0.48 | +0.40 | +0.50 | +0.49 | +0.40 | +0.51 | +0.54 | +0.17 | +0.33 |
| Pre-industrial · ERA5 SST | +0.31 | — | +0.35 | +0.24 | +0.36 | +0.40 | +0.25 | +0.36 | +0.40 | +0.09 | +0.10 |
| Pre-industrial · ERA5/GFS air | +0.47 | +0.45 | +0.47 | +0.46 | +0.48 | +0.47 | +0.33 | +0.47 | +0.49 | +0.41 | +0.08 |

(Positive offsets raise the curve. A 1991–2020 offset is negative where 1991–2020
was warmer than 1971–2000. The OISST and ERA5 SST rows differ because their
1971–2000 climatologies differ (§3.1). Their anomalies *against pre-industrial*
are the comparable quantities.)

## 5. Caveats

- **Pre-industrial is an estimate.** Observations in 1850–1900 were sparse,
  especially in the Southern Hemisphere, the polar oceans and continental
  interiors. Values there come largely from the reconstructions' statistical
  infilling and from our band fill. The Arctic and Southern Ocean offsets are the
  least certain.
- **Datasets disagree.** Each reconstruction's 1991–2020 → 1850–1900 step differs
  from the others by up to about 0.1 °C globally, and by more regionally. Averaging
  them reduces, but does not remove, that spread.
- **Early SST is biased.** Ship-based measurements carry known biases. Sippel et
  al. (2024) find early-20th-century SST reconstructions too cold by roughly
  0.26 °C around 1900–1930, while concluding that long-term warming since the
  mid-19th century is unaffected.
- **Blended records over the ocean.** The air-temperature step over open ocean
  comes from SST-based records (§3.2, within 10% per AR6).
- **Daily maximum and minimum.** GFS daily-max and daily-min anomalies reuse the
  daily-mean offset. No reconstruction resolves the day–night range back to 1850.
- **Smooth in space and season.** Offsets vary on scales of about 500 km and
  follow a smooth annual cycle. All finer detail comes from the modern data.
- **ERA5 SST changes source in 2007.** ERA5's SST input switches from HADISST2 to
  OSTIA in September 2007, which adds a small step within 1991–2020 that its
  offset inherits.
- **Regional means are exact only on matching cells.** Each source's regional
  offset is averaged over exactly the cells its daily data covers (OISST's fixed
  land mask, ERA5's SST land mask, or every cell for air temperature), so the
  identity in §1 holds for the graphs.

## 6. Files and reproduction

```
uv run scripts/build_baseline_deltas.py          # 1991-2020, then pre-industrial, then offsets + atlases
uv run scripts/build_baseline_deltas.py --offsets-only   # e.g. after adding a region
uv run upload-to-s3.py --maps-dir ./maps                 # publish the atlases
```

- `baselines/fields/delta-{sst,t2m,oisst}-{1991-2020,preindustrial}.nc`: Δ
  per calendar month on the 0.25° OISST grid, in °C, to be added to a 1971–2000
  anomaly. `sst`/`t2m` apply to ERA5 and GFS; `oisst` to OISST.
- `baselines/offsets.json` (committed): the seasonal coefficients
  `[a0, a1, b1, a2, b2]` of each field's area-weighted regional mean,
  `{baseline: {source: {dataset: {region: [...]}}}}`. `export_timeseries.py`
  copies each series' entry into its region JSON as `baseline_offsets`.
- `maps/baselines/delta-{field}-{baseline}.webp` and `manifest.json`, published
  to `sea-surface-temp/baselines/`: the per-cell coefficients for the globe. Each
  is a lossless 1440×3600 WebP of five north-up 0.25° tiles ($a_0$ on top). Each
  pixel holds a 16-bit count in R (high byte) and G (low byte):
  value = −32.768 + 0.002 × count °C. That keeps any day's $\Delta$ within
  0.005 °C of the full field. The manifest gives the encoding, the tile layout,
  and which atlas applies to each (baseline, source, anomaly dataset).
- `baselines/raw/`: cached downloads (about 5 GB).

The build takes a few minutes once the downloads are cached. ERA5 monthly means
need a CDS account (`~/.cdsapirc`). Everything else is public.

## References

- Copernicus Climate Change Service: [Climate Bulletin: about the data and analysis](https://climate.copernicus.eu/climate-bulletin-about-data-and-analysis) and [Temperature Q&As](https://climate.copernicus.eu/temperature-qas): the pre-industrial offset method for ERA5.
- Gulev, S. K., et al. (2021). Changing state of the climate system. In *Climate Change 2021: The Physical Science Basis* (IPCC AR6 WGI), Chapter 2 and Cross-Chapter Box 2.3.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146, 1999–2049.
- Hirahara, S., Ishii, M., Fukuda, Y. (2014). Centennial-scale sea surface temperature analysis and its uncertainty. *J. Climate* 27, 57–75 (COBE-SST2).
- Huang, B., et al. (2021). Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1. *J. Climate* 34, 2923–2939.
- Kennedy, J. J., et al. (2019). An ensemble data set of sea surface temperature change from 1850: the Met Office Hadley Centre HadSST.4.0.0.0 data set. *J. Geophys. Res. Atmos.* 124, 7719–7763.
- Morice, C. P., et al. (2021). An updated assessment of near-surface temperature change from 1850: the HadCRUT5 data set. *J. Geophys. Res. Atmos.* 126, e2019JD032361.
- NOAA NCEI: [ERSST v6](https://www.ncei.noaa.gov/products/extended-reconstructed-sst) and [NOAAGlobalTemp v6.1](https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp) product documentation.
- Rohde, R. A., Hausfather, Z. (2020). The Berkeley Earth Land/Ocean Temperature Record. *Earth Syst. Sci. Data* 12, 3469–3479.
- Sippel, S., et al. (2024). Early-twentieth-century cold bias in ocean surface temperature observations. *Nature* 635, 618–624. [doi:10.1038/s41586-024-08230-1](https://doi.org/10.1038/s41586-024-08230-1)
