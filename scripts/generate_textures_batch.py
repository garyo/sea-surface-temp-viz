#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Bulk-generate globe textures from a source's local NetCDF archive.

The daily CI cron only renders textures for the most recent few days; this
script fills the back-catalog for a given date range using already-present
archive files (``era5-archive/`` or OISST's ``netcdf-archive/`` — pure local
CPU work, no CDS/GCS/NCEI traffic) and writes the same textures + metadata
that ``pipeline.py --mode texture`` produces (see textures.write_textures).

Why a batch script and not a loop over ``pipeline.py``? Each ``pipeline.py``
invocation pays ~2–3 s of matplotlib/xarray import overhead before doing
~0.1 s of real work; for 1700+ files that's an hour of pure cold-start.
This script imports once and processes everything in-process.

Outputs land in ``./maps/`` (matches the CI convention), so the existing
``upload-to-s3.py`` can push them and regenerate ``index.json``.

Usage:
    uv run scripts/generate_textures_batch.py [--source era5] [--start 2024-01-01] [--skip-existing-on-s3]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sources import DataSource, Era5Source, OisstSource
from textures import write_textures

# Sources whose days live in a local archive (GFS fetches its own; see
# backfill_gfs.py).
ARCHIVED_SOURCES: dict[str, type[DataSource]] = {
    s.id: s for s in (Era5Source, OisstSource)
}

DEFAULT_START = datetime.date(2024, 1, 1)
S3_INDEX_URL = (
    "https://climate-change-assets.s3.amazonaws.com/sea-surface-temp/index.json"
)


def fetch_s3_dates(source_id: str) -> set[str]:
    """Return the set of date strings (YYYY-MM-DD) that already have at
    least one texture for this source on S3, per index.json. We treat
    presence in the index as "every dataset exists" — the CI cron always
    renders them together, so this is a safe coarse-grained filter.
    """
    try:
        with urllib.request.urlopen(S3_INDEX_URL, timeout=30) as resp:
            data = json.load(resp)
        return set(data.get("sources", {}).get(source_id, {}).get("dates", []))
    except Exception as e:  # noqa: BLE001
        print(f"⚠  Failed to fetch S3 index ({e}); not skipping anything")
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(ARCHIVED_SOURCES), default="era5")
    parser.add_argument(
        "--archive-root", type=Path, help="Defaults to the source's archive_root"
    )
    parser.add_argument("--out", type=Path, default=Path("./maps"))
    parser.add_argument(
        "--start",
        type=datetime.date.fromisoformat,
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        type=datetime.date.fromisoformat,
        default=datetime.datetime.now(datetime.UTC).date() - datetime.timedelta(days=7),
    )
    parser.add_argument(
        "--skip-existing-on-s3",
        action="store_true",
        default=True,
        help="Skip dates already present in the S3 index (default)",
    )
    parser.add_argument(
        "--no-skip-existing-on-s3",
        dest="skip_existing_on_s3",
        action="store_false",
        help="Regenerate everything regardless of S3 state",
    )
    parser.add_argument(
        "--datasets",
        type=lambda s: [d.strip() for d in s.split(",") if d.strip()],
        default=None,
        help="Comma-separated dataset ids to render (e.g. sst_anom,t2m_anom). "
        "Defaults to all of the source's datasets. Use this to backfill only "
        "the anomaly variants, which were added after the SST/T2M textures and "
        "so are missing from the historical back-catalog. NOTE: the S3 index "
        "marks a date as present once *any* era5 texture exists for it, so "
        "--skip-existing-on-s3 would skip those dates even though their anomaly "
        "texture is missing — pair --datasets with --no-skip-existing-on-s3.",
    )
    args = parser.parse_args(argv)
    source_cls = ARCHIVED_SOURCES[args.source]
    archive_root = args.archive_root or source_cls.archive_root

    args.out.mkdir(parents=True, exist_ok=True)

    skip_dates: set[str] = set()
    if args.skip_existing_on_s3:
        skip_dates = fetch_s3_dates(args.source)
        print(f"Existing on S3 to skip: {len(skip_dates)} dates")

    source = source_cls()
    dataset_ids = args.datasets or list(source.datasets)
    unknown = [d for d in dataset_ids if d not in source.datasets]
    if unknown:
        parser.error(
            f"unknown dataset(s) {unknown}; {args.source} has {list(source.datasets)}"
        )
    print(f"Datasets: {dataset_ids}")

    one = datetime.timedelta(days=1)
    dates: list[datetime.date] = []
    d = args.start
    while d <= args.end:
        dates.append(d)
        d += one

    print(f"Date range: {args.start} → {args.end} ({len(dates)} days)")
    print(f"Output dir: {args.out.resolve()}")
    print()

    started = time.time()
    ok = skip = miss = fail = 0
    for i, d in enumerate(dates, 1):
        date_str = d.isoformat()
        if date_str in skip_dates:
            skip += 1
            continue
        nc_path = source_cls.archive_path(archive_root, d)
        if not nc_path.exists():
            miss += 1
            print(f"  ⚠  missing archive {nc_path.name}")
            continue
        try:
            with source.open_local(nc_path) as raw:
                for ds_id in dataset_ids:
                    data = source.get_data_array(raw, ds_id)
                    write_textures(source, ds_id, d, data, args.out)
            ok += 1
        except Exception as e:  # noqa: BLE001 — surface for triage
            fail += 1
            print(f"  ❌ {date_str}: {type(e).__name__}: {e}")

        if i % 50 == 0 or i == len(dates):
            elapsed = time.time() - started
            rate = i / max(elapsed, 1)
            eta = (len(dates) - i) / max(rate, 0.001)
            print(
                f"[{i}/{len(dates)}] ok={ok} skip={skip} miss={miss} fail={fail} "
                f"({rate:.1f}/s, ETA {eta / 60:.1f}m)"
            )

    print()
    print(
        f"Done. Wrote: {ok}, Skipped: {skip}, Missing archive: {miss}, Failed: {fail}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
