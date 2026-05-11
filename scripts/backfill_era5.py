#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Download an ERA5 single-levels archive (SST + 2 m air temp) for backfill.

One CDS request per day at 12:00 UTC, resampled to the OISST 720×1440 grid
at fetch time and stored compressed as ``./era5-archive/YYYY/era5-YYYYMMDD.nc``
(~3 MB/day after zlib level 4).

Resumable — files that already exist are skipped, so you can interrupt with
Ctrl-C and re-run. Concurrency is bounded by the CDS server itself (typically
6 simultaneous requests per user); going wider here just queues earlier.

Usage:
    uv run scripts/backfill_era5.py [--start 2021-01-01] [--workers 6]
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow the sibling sources package to be imported when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sources.era5 import Era5Source, Era5FetchError, _do_cds_retrieve  # noqa: E402

# ERA5 begins 1940 but we match OISST's start date for cross-source comparability.
DEFAULT_START = datetime.date(2021, 1, 1)


def date_range(start: datetime.date, end: datetime.date):
    d = start
    one = datetime.timedelta(days=1)
    while d <= end:
        yield d
        d += one


def download_one(
    d: datetime.date, archive_root: Path
) -> tuple[datetime.date, str]:
    """Returns (date, status) where status is 'ok', 'skip', or 'fail: <reason>'."""
    out = Era5Source.archive_path(archive_root, d)
    if out.exists() and out.stat().st_size > 0:
        return (d, "skip")
    try:
        _do_cds_retrieve(d, out)
        return (d, "ok")
    except Era5FetchError as e:
        return (d, f"fail: {e}")
    except Exception as e:  # noqa: BLE001 — surface the underlying error for triage
        return (d, f"fail: {type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Era5Source.archive_root,
        help="Where to save the resampled NetCDF archive",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.date.today() - datetime.timedelta(days=7),
        help="Inclusive end date (default: 7 days before today, past ERA5T latency)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent CDS requests (CDS itself caps each user around 6)",
    )
    args = parser.parse_args(argv)

    args.archive_root.mkdir(parents=True, exist_ok=True)
    dates = list(date_range(args.start, args.end))
    print(f"Backfilling ERA5: {args.start} → {args.end} ({len(dates)} days)")
    print(f"Archive root: {args.archive_root.resolve()}")
    print(f"Workers: {args.workers}")
    print()

    started = time.time()
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, d, args.archive_root) for d in dates]
        for i, f in enumerate(as_completed(futures), 1):
            d, status = f.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  ❌ {d}: {status}")
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - started
                rate = i / max(elapsed, 1)
                eta = (len(futures) - i) / max(rate, 0.001)
                print(
                    f"[{i}/{len(futures)}] ok={ok} skip={skip} fail={fail} "
                    f"({rate:.2f}/s, ETA {eta/60:.1f}m)"
                )

    print()
    print(f"Done. Downloaded: {ok}, Already had: {skip}, Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
