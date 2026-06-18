#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Compute regional aggregates from a local source archive into data-cache.json.

Walks ``./<archive-root>/YYYY/*.<ext>`` (created by the per-source backfill
scripts) and, for each file, computes cosine-weighted aggregates for every
region defined in ``regions.py`` for every dataset the source exposes, then
writes them to ``data-cache.json`` under keys
``YYYY-MM-DD-{source}-{dataset}-{region}``.

Already-cached entries are not recomputed. Resumable: interrupt with Ctrl-C
and re-run; a checkpoint is flushed every ``--flush-every`` files (default 50).

Usage:
    uv run scripts/aggregate_archive.py
    uv run scripts/aggregate_archive.py --workers 4 --flush-every 100
    uv run scripts/aggregate_archive.py --regions global  # refresh one region
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# Allow the sibling regions.py / sources package to be imported when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import regions  # noqa: E402
from sources import SOURCES  # noqa: E402

# Plausibility bounds for a regional daily-mean value (°C). A corrupt or
# upstream-flagged NetCDF that still parses (e.g. ECMWF's "should not be used"
# daily-statistics files) can yield wildly out-of-range means; without this
# guard those would now be persisted to the S3 cache and shown forever. A whole
# date is rejected (raised, hence retried on a later run) rather than writing a
# partial/poisoned entry. Absolute temps span roughly -90..+60 °C; regional
# anomalies realistically stay well inside ±30 °C.
_PLAUSIBLE_RANGE = {
    "abs": (-100.0, 70.0),    # sst, t2m
    "anom": (-40.0, 40.0),    # sst_anom, t2m_anom
}


def _check_plausible(ds_id: str, region_id: str, val: float) -> None:
    import math

    if math.isnan(val):
        return  # masked/empty region — legitimately NaN, handled downstream
    lo, hi = _PLAUSIBLE_RANGE["anom" if ds_id.endswith("anom") else "abs"]
    if not (lo <= val <= hi):
        raise ValueError(
            f"implausible {ds_id} value {val:.3f}°C for {region_id} "
            f"(outside [{lo}, {hi}]) — likely corrupt/flagged source file"
        )


def aggregate_one(
    nc_path: Path,
    source_id: str,
    region_ids: list[str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Returns ``(date_str, {cache_key: value, ...})`` for the requested
    regions × every dataset the source exposes. If ``region_ids`` is None,
    all regions defined in regions.py are computed.
    """
    source = SOURCES[source_id]()
    ymd = source.date_from_filename(nc_path)
    if ymd is None:
        raise ValueError(f"Unrecognized filename for source {source_id}: {nc_path}")
    y, m, d = ymd
    date_str = f"{y:04}-{m:02}-{d:02}"
    regs = region_ids if region_ids is not None else regions.region_ids()

    out: dict[str, float] = {}
    with source.open_local(nc_path) as raw:
        lat_2d, lon_2d = source.latlon_2d(raw)
        for ds_id in source.datasets:
            data = source.get_data_array(raw, ds_id)
            for region_id in regs:
                val = regions.aggregate(data, lat_2d, lon_2d, region_id)
                _check_plausible(ds_id, region_id, val)
                key = f"{date_str}-{source.id}-{ds_id}-{region_id}"
                out[key] = val
    return (date_str, out)


def all_keys_present(
    date_str: str,
    cache: dict[str, float],
    region_ids: list[str],
    source_id: str,
    dataset_ids: list[str],
) -> bool:
    """All region × dataset keys for this date already cached?"""
    for ds in dataset_ids:
        for r in region_ids:
            if f"{date_str}-{source_id}-{ds}-{r}" not in cache:
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
    parser.add_argument("--source", default="oisst", choices=sorted(SOURCES.keys()))
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Local NetCDF archive (default: source-specific, e.g. ./netcdf-archive for oisst)",
    )
    parser.add_argument("--cache-file", type=Path, default=Path("./data-cache.json"))
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel processes (each opens one NetCDF at a time)")
    parser.add_argument("--flush-every", type=int, default=50,
                        help="Save data-cache.json every N processed files")
    parser.add_argument(
        "--regions", type=str, default="",
        help="Comma-separated region ids to recompute (default: all). "
             "When set, every NetCDF in the archive is processed (the "
             "all-keys-present skip is bypassed) and only the listed "
             "regions are written to the cache. Useful for refreshing a "
             "single basin after a mask change.",
    )
    args = parser.parse_args(argv)

    source_cls = SOURCES[args.source]
    dataset_ids = list(source_cls.datasets.keys())
    archive_root: Path = args.archive_root or source_cls.archive_root
    args.archive_root = archive_root  # also feeds the existence check below

    if args.regions:
        region_ids = [r.strip() for r in args.regions.split(",") if r.strip()]
        unknown = [r for r in region_ids if r not in regions.region_ids()]
        if unknown:
            print(f"❌ Unknown regions: {unknown}", file=sys.stderr)
            print(f"Available: {regions.region_ids()}", file=sys.stderr)
            return 1
        print(f"Restricting recomputation to: {region_ids}")
    else:
        region_ids = regions.region_ids()

    if not args.archive_root.is_dir():
        print(f"❌ Archive not found: {args.archive_root}", file=sys.stderr)
        return 1

    cache = load_cache(args.cache_file)
    print(f"Loaded {len(cache)} existing cache entries from {args.cache_file}")

    nc_files = sorted(args.archive_root.rglob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files in {args.archive_root}")

    if args.regions:
        # Targeted refresh: re-process every file, overwriting only the
        # requested regions. The skip check would otherwise mask the fact
        # that the existing cache entries are stale (wrong mask).
        todo = [nc for nc in nc_files if source_cls.date_from_filename(nc) is not None]
        print(f"To compute: {len(todo)} files (all, --regions overrides skip)")
    else:
        todo = []
        for nc in nc_files:
            ymd = source_cls.date_from_filename(nc)
            if ymd is None:
                continue
            date_str = f"{ymd[0]:04}-{ymd[1]:02}-{ymd[2]:02}"
            if not all_keys_present(date_str, cache, region_ids, args.source, dataset_ids):
                todo.append(nc)
        print(f"To compute: {len(todo)} files (skipping {len(nc_files) - len(todo)} fully cached)")

    if not todo:
        print("Nothing to do.")
        return 0

    started = time.time()
    n_done = 0
    aggregate_fn = partial(aggregate_one, source_id=args.source, region_ids=region_ids)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(aggregate_fn, nc): nc for nc in todo}
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
