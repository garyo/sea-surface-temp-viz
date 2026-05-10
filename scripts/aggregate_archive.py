#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy"]
# ///
# SPDX-License-Identifier: MIT
"""Compute regional aggregates from a local OISST NetCDF archive into data-cache.json.

Walks ./netcdf-archive/YYYY/*.nc (created by scripts/backfill_oisst.py) and,
for each file, computes cosine-weighted aggregates for every region defined in
regions.py for both the SST and anomaly datasets, then writes them to
data-cache.json under keys ``YYYY-MM-DD-oisst-{sst|anom}-{region}``.

Already-cached entries are not recomputed. Resumable: interrupt with Ctrl-C
and re-run; a checkpoint is flushed every --flush-every files (default 50).

Usage:
    uv run scripts/aggregate_archive.py
    uv run scripts/aggregate_archive.py --workers 4 --flush-every 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

# Allow the sibling regions.py to be imported when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import regions  # noqa: E402

NC_NAME_RE = re.compile(r"oisst-avhrr-v02r01\.(\d{4})(\d{2})(\d{2})(?:_preliminary)?\.nc$")


def date_from_name(path: Path) -> tuple[int, int, int] | None:
    m = NC_NAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_processed_array(hdf, dataset_name: str) -> np.ma.MaskedArray:
    """Mirror sea-surface-temps.get_processed_hdf_data_array (no lat/ice mask)."""
    arr = hdf[dataset_name][0][0][:][:]
    scale_factor = hdf[dataset_name].attrs["scale_factor"]
    assert hdf[dataset_name].attrs["add_offset"] == 0
    land_mask = arr == -999
    masked = np.ma.array(arr, mask=land_mask)
    return masked * scale_factor


def latlon_2d(hdf):
    lat = hdf["lat"][:]
    lon = hdf["lon"][:]
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    return lat_2d, lon_2d


def aggregate_one(nc_path: Path) -> tuple[str, dict[str, float]]:
    """Returns (date_str, {cache_key: value, ...}) for all regions × {sst, anom}."""
    ymd = date_from_name(nc_path)
    if ymd is None:
        raise ValueError(f"Unrecognized filename: {nc_path}")
    y, m, d = ymd
    date_str = f"{y:04}-{m:02}-{d:02}"

    with h5py.File(nc_path, "r") as hdf:
        lat_2d, lon_2d = latlon_2d(hdf)
        out: dict[str, float] = {}
        for ds_name in ("sst", "anom"):
            data = get_processed_array(hdf, ds_name)
            for region_id in regions.region_ids():
                val = regions.aggregate(data, lat_2d, lon_2d, region_id)
                key = f"{date_str}-oisst-{ds_name}-{region_id}"
                out[key] = val
    return (date_str, out)


def all_keys_present(date_str: str, cache: dict[str, float]) -> bool:
    """All region × dataset keys for this date already cached?"""
    for ds in ("sst", "anom"):
        for r in regions.region_ids():
            if f"{date_str}-oisst-{ds}-{r}" not in cache:
                return False
    return True


def load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def save_cache(path: Path, cache: dict[str, float]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as f:
        json.dump(cache, f, sort_keys=True, indent=2)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("./netcdf-archive"))
    parser.add_argument("--cache-file", type=Path, default=Path("./data-cache.json"))
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel processes (each opens one NetCDF at a time)")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Save data-cache.json every N processed files")
    args = parser.parse_args(argv)

    if not args.archive_root.is_dir():
        print(f"❌ Archive not found: {args.archive_root}", file=sys.stderr)
        return 1

    cache = load_cache(args.cache_file)
    print(f"Loaded {len(cache)} existing cache entries from {args.cache_file}")

    nc_files = sorted(args.archive_root.rglob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files in {args.archive_root}")

    todo: list[Path] = []
    for nc in nc_files:
        ymd = date_from_name(nc)
        if ymd is None:
            continue
        date_str = f"{ymd[0]:04}-{ymd[1]:02}-{ymd[2]:02}"
        if not all_keys_present(date_str, cache):
            todo.append(nc)
    print(f"To compute: {len(todo)} files (skipping {len(nc_files) - len(todo)} fully cached)")

    if not todo:
        print("Nothing to do.")
        return 0

    started = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(aggregate_one, nc): nc for nc in todo}
        for fut in as_completed(futures):
            try:
                date_str, entries = fut.result()
            except Exception as e:
                nc = futures[fut]
                print(f"  ❌ {nc.name}: {e}")
                continue
            cache.update(entries)
            n_done += 1
            if n_done % args.flush_every == 0:
                save_cache(args.cache_file, cache)
            if n_done % 100 == 0 or n_done == len(todo):
                elapsed = time.time() - started
                rate = n_done / max(elapsed, 1)
                eta = (len(todo) - n_done) / max(rate, 0.001)
                print(
                    f"[{n_done}/{len(todo)}] {date_str} "
                    f"({rate:.1f}/s, ETA {eta/60:.1f}m)"
                )

    save_cache(args.cache_file, cache)
    print(f"Wrote {args.cache_file} ({len(cache)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
