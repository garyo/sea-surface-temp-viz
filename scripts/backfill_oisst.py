#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["requests", "urllib3"]
# ///
# SPDX-License-Identifier: MIT
"""Download the entire OISST v2.1 daily NetCDF archive to a local folder.

This is a ONE-TIME setup tool — run on a laptop overnight. The archive (~24GB,
~16K files from 1982-09 to present at the time of writing) lets you compute
regional aggregates for any region defined in regions.py without needing to
re-download. Once the archive exists, scripts/aggregate_archive.py does the
actual aggregation in ~30 min for all regions.

Files are saved as ``./netcdf-archive/YYYY/oisst-avhrr-v02r01.YYYYMMDD.nc``.
Existing files are skipped, so the script is resumable. Run with --workers N
to tune concurrency (default 8; OISST throughput is variable).

Usage:
    uv run scripts/backfill_oisst.py [--start 1982-09-01] [--workers 8]
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL_BASE = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr"
)

# OISST v2.1 begins 1981-09-01, but the existing pipeline used 1982 as start.
# Use 1982-01-01 to match.
DEFAULT_START = datetime.date(1982, 1, 1)


def date_range(start: datetime.date, end: datetime.date):
    d = start
    one = datetime.timedelta(days=1)
    while d <= end:
        yield d
        d += one


def file_path(archive_root: Path, d: datetime.date) -> Path:
    return archive_root / f"{d.year:04}" / f"oisst-avhrr-v02r01.{d.year:04}{d.month:02}{d.day:02}.nc"


def urls_for(d: datetime.date) -> list[str]:
    ym = f"{d.year:04}{d.month:02}"
    ymd = f"{d.year:04}{d.month:02}{d.day:02}"
    return [
        f"{URL_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}.nc",
        f"{URL_BASE}/{ym}/oisst-avhrr-v02r01.{ymd}_preliminary.nc",
    ]


def download_one(d: datetime.date, archive_root: Path, retries: int = 3) -> tuple[datetime.date, str]:
    """Returns (date, status) where status is 'ok', 'skip', or 'fail: <reason>'."""
    out = file_path(archive_root, d)
    if out.exists() and out.stat().st_size > 0:
        return (d, "skip")
    out.parent.mkdir(parents=True, exist_ok=True)

    last_err = ""
    for url in urls_for(d):
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=60, verify=False)
                if resp.status_code == 200 and len(resp.content) > 0:
                    tmp = out.with_suffix(out.suffix + ".part")
                    tmp.write_bytes(resp.content)
                    tmp.rename(out)
                    return (d, "ok")
                last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = str(e)
                time.sleep(1 + attempt)  # gentle backoff
    return (d, f"fail: {last_err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("./netcdf-archive"),
        help="Where to save the NetCDF archive",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
        default=datetime.date.today() - datetime.timedelta(days=2),
        help="Inclusive end date (default: 2 days before today)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    args.archive_root.mkdir(parents=True, exist_ok=True)
    dates = list(date_range(args.start, args.end))
    print(f"Backfilling OISST: {args.start} → {args.end} ({len(dates)} days)")
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
            if i % 100 == 0 or i == len(futures):
                elapsed = time.time() - started
                rate = i / max(elapsed, 1)
                eta = (len(futures) - i) / max(rate, 0.001)
                print(
                    f"[{i}/{len(futures)}] ok={ok} skip={skip} fail={fail} "
                    f"({rate:.1f}/s, ETA {eta/60:.1f}m)"
                )

    print()
    print(f"Done. Downloaded: {ok}, Already had: {skip}, Failed: {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
