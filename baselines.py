# SPDX-License-Identifier: MIT
"""Alternative anomaly baselines, expressed as additive offsets.

Every anomaly the pipeline publishes is relative to 1971–2000 (OISST's own
climatology, or our ERA5 daily climatology for ERA5 and GFS). Moving to another
baseline B doesn't need new climatologies at daily resolution, because the
difference is additive and the same every year:

    anomaly_vs_B(cell, t) = anomaly_vs_1971–2000(cell, t) + Δ_B(cell, τ(t))
    Δ_B = mean over 1971–2000 − mean over B

Δ_B is built per calendar month on the OISST 720×1440 grid
(``fields/delta-{field}-{baseline}.nc``, by ``scripts/build_baseline_deltas.py``)
and reduced to a mean plus two annual harmonics, so everything downstream
carries five coefficients per cell (or region) and evaluates

    Δ(τ) = a0 + a1·cos 2πτ + b1·sin 2πτ + a2·cos 4πτ + b2·sin 4πτ

at the fraction of the year τ — exactly, on any day, with no month
interpolation. Month m's mean sits at τ = (m − ½)/12; the site uses
τ = (day-of-year + ½)/366 on its leap-aligned day axis. Region coefficients —
what the time-series graphs need — are small enough to commit as
``offsets.json``, so the nightly export never touches the large fields.

A regional mean of a sum is the sum of regional means only over the *same*
cells, so each source's offsets are averaged over that source's own valid cells
(OISST's land mask, ERA5's SST land mask, or everywhere for air temperature).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import regions

BASELINE_DIR = Path(__file__).parent / "baselines"
FIELD_DIR = BASELINE_DIR / "fields"
OFFSETS_PATH = BASELINE_DIR / "offsets.json"

REFERENCE_YEARS = (1971, 2000)


@dataclass(frozen=True)
class Baseline:
    id: str
    label: str
    years: tuple[int, int]


BASELINES: dict[str, Baseline] = {
    b.id: b
    for b in (
        Baseline("1991-2020", "1991–2020", (1991, 2020)),
        Baseline("preindustrial", "pre-industrial (1850–1900)", (1850, 1900)),
    )
}

# Which physical variable's Δ applies to each published anomaly dataset. GFS
# daily max/min anomalies reuse the daily-mean Δ: no reconstruction resolves
# the diurnal range back to 1850, and its change is small next to the mean's.
ANOMALY_VARIABLE: dict[tuple[str, str], str] = {
    ("oisst", "anom"): "sst",
    ("era5", "sst_anom"): "sst",
    ("era5", "t2m_anom"): "t2m",
    ("gfs", "t2m_mean_anom"): "t2m",
    ("gfs", "t2m_max_anom"): "t2m",
    ("gfs", "t2m_min_anom"): "t2m",
}

# OISST publishes daily climatologies for both 1971–2000 (what its anomalies
# use) and 1991–2020, so its 1991–2020 Δ comes from those rather than ERA5 —
# matching how others (e.g. Climate Reanalyzer) show OISST vs 1991–2020 — and
# its pre-industrial Δ builds on that one (see build_baseline_deltas.py).
FIELD_OVERRIDES: dict[tuple[str, str], str] = {
    ("1991-2020", "oisst"): "oisst",
    ("preindustrial", "oisst"): "oisst",
}


def delta_field(baseline_id: str, source: str, dataset: str) -> str:
    """Which Δ field (``sst``, ``t2m``, or a source-specific one) applies."""
    return FIELD_OVERRIDES.get(
        (baseline_id, source), ANOMALY_VARIABLE[(source, dataset)]
    )


# Valid-cell mask each source's regional means are taken over, matching what
# regions.aggregate() sees in the daily data. None = every cell.
SOURCE_MASK: dict[tuple[str, str], str | None] = {
    ("oisst", "sst"): "valid-oisst-sst.npy",
    ("era5", "sst"): "valid-era5-sst.npy",
    ("era5", "t2m"): None,
    ("gfs", "t2m"): None,
}

OffsetTable = dict[str, dict[str, dict[str, dict[str, list[float]]]]]
"""baseline → source → dataset → region → [a0, a1, b1, a2, b2] (°C)."""

SEASONAL_HARMONICS = 2


def seasonal_basis(tau: np.ndarray) -> np.ndarray:
    """(len(tau), 5) design matrix: 1, cos 2πτ, sin 2πτ, cos 4πτ, sin 4πτ."""
    cols = [np.ones_like(tau)]
    for j in range(1, SEASONAL_HARMONICS + 1):
        cols += [np.cos(2 * np.pi * j * tau), np.sin(2 * np.pi * j * tau)]
    return np.stack(cols, axis=1)


MONTH_TAU = (np.arange(12) + 0.5) / 12


def seasonal_coefficients(monthly: np.ndarray) -> np.ndarray:
    """Least-squares [a0, a1, b1, a2, b2] (leading axis) for 12 monthly values.

    Δ fields are already reduced to these harmonics, so the fit is exact.
    """
    coefficients, *_ = np.linalg.lstsq(
        seasonal_basis(MONTH_TAU), monthly.reshape(12, -1), rcond=None
    )
    return coefficients.reshape(-1, *monthly.shape[1:])


def delta_path(baseline_id: str, field: str) -> Path:
    return FIELD_DIR / f"delta-{field}-{baseline_id}.nc"


def load_delta(baseline_id: str, field: str) -> np.ndarray:
    """Return Δ as a (12, 720, 1440) float32 array in °C."""
    import xarray as xr

    path = delta_path(baseline_id, field)
    if not path.exists():
        raise FileNotFoundError(
            f"No Δ field at {path}. Run scripts/build_baseline_deltas.py "
            f"--baseline {baseline_id}."
        )
    with xr.open_dataset(path) as ds:
        return np.asarray(ds["delta"].values, dtype=np.float32)


def load_valid_mask(source: str, variable: str) -> np.ndarray | None:
    name = SOURCE_MASK[(source, variable)]
    if name is None:
        return None
    path = FIELD_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"No valid-cell mask at {path}. Run scripts/build_baseline_deltas.py."
        )
    return np.load(path)


def regional_offsets(
    delta: np.ndarray,
    valid: np.ndarray | None,
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
) -> dict[str, list[float]]:
    """Each region's seasonal coefficients of its cos-lat-weighted mean Δ."""
    weights = np.cos(np.radians(lat_2d))
    out: dict[str, list[float]] = {}
    for rid in regions.region_ids():
        cells = regions.contains(rid, lat_2d, lon_2d)
        if valid is not None:
            cells = cells & valid
        w = weights[cells]
        monthly = np.array([np.average(month[cells], weights=w) for month in delta])
        out[rid] = [round(float(c), 4) for c in seasonal_coefficients(monthly)]
    return out


def load_offsets(path: Path = OFFSETS_PATH) -> OffsetTable:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def offsets_for(
    table: OffsetTable, source: str, dataset: str, region: str
) -> dict[str, list[float]]:
    """``{baseline: [a0, a1, b1, a2, b2]}`` for one series; empty if none apply."""
    return {
        baseline: by_source[source][dataset][region]
        for baseline, by_source in table.items()
        if region in by_source.get(source, {}).get(dataset, {})
    }
