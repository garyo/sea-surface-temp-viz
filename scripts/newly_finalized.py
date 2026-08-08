#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Find dates whose data just flipped from preliminary to final.

NOAA publishes OISST ``_preliminary`` files for ~2 weeks and then replaces them.
The daily workflow renders textures for a fixed recent window, so a day rendered
while it was still preliminary would keep its provisional imagery forever. The
``{date}-{source}-preliminary-flag`` cache keys written by pipeline.py let us
spot exactly which days settled during this run and re-render only those,
instead of blindly re-rendering the whole finalization window every night.

Two-phase, around the prune + re-fetch that the graph step performs:

    # before prune-cache.py
    uv run scripts/newly_finalized.py --snapshot-out /tmp/prelim-before.json

    # after the graph runs have re-fetched the pruned window
    EXTRA_DAYS=$(uv run scripts/newly_finalized.py --snapshot-in /tmp/prelim-before.json)

The second form prints one ``--days-ago`` integer per line (that is the only
date selector pipeline.py accepts), ready to append to the texture loop.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

FLAG_SUFFIX = "-preliminary-flag"


def flagged_dates(cache_path: Path, source: str) -> set[str]:
    """Dates currently marked preliminary for ``source``."""
    with cache_path.open("r") as f:
        cache = json.load(f)
    suffix = f"-{source}{FLAG_SUFFIX}"
    return {k[:10] for k in cache if k.endswith(suffix)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-file", type=Path, default=Path("./data-cache.json"))
    parser.add_argument("--source", default="oisst")
    parser.add_argument(
        "--snapshot-out",
        type=Path,
        help="Write the currently-flagged dates here and exit",
    )
    parser.add_argument(
        "--snapshot-in",
        type=Path,
        help="Compare against this snapshot and print newly-final days-ago values",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Ignore dates older than this many days — a far-past flag clearing "
        "is a cache repair, not a fresh finalization worth re-rendering",
    )
    args = parser.parse_args(argv)

    if not args.cache_file.exists():
        print(f"❌ Cache file not found: {args.cache_file}", file=sys.stderr)
        return 1

    if args.snapshot_out:
        dates = sorted(flagged_dates(args.cache_file, args.source))
        with args.snapshot_out.open("w") as f:
            json.dump(dates, f)
        print(f"📌 {len(dates)} {args.source} date(s) currently preliminary")
        return 0

    if not args.snapshot_in:
        parser.error("one of --snapshot-out or --snapshot-in is required")

    if not args.snapshot_in.exists():
        # First run after this landed, or a lost temp file. Nothing to compare
        # against, so re-render nothing extra rather than guessing.
        print(
            f"ℹ️  No snapshot at {args.snapshot_in}; nothing to re-render",
            file=sys.stderr,
        )
        return 0

    with args.snapshot_in.open("r") as f:
        before = set(json.load(f))
    finalized = before - flagged_dates(args.cache_file, args.source)

    today = datetime.datetime.now(datetime.UTC).date()
    days_ago = sorted(
        age
        for d in finalized
        if 0 < (age := (today - datetime.date.fromisoformat(d)).days) <= args.max_age
    )
    print(
        f"✅ {len(days_ago)} {args.source} date(s) finalized this run", file=sys.stderr
    )
    for age in days_ago:
        print(age)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
