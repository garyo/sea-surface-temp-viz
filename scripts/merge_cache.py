#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Merge the S3-persisted data cache onto the committed one.

The committed ``data-cache.json`` is the authority for historical dates (it is
hand-curated by deliberate backfill commits and frozen well in the past). The
S3 cache is what CI accumulates run-to-run for *recent* dates, which are never
committed back. We want, for every key:

    prefer the committed value if present (authoritative / corrected history),
    otherwise take the S3 value (the recent tail the commit doesn't have yet).

That is exactly ``dict(s3); update(committed)`` — committed wins on every shared
key, and S3 only contributes keys the commit lacks. The result is written to
``--out`` (defaulting to the committed file, i.e. an in-place upgrade).

Missing or unreadable S3 file is not an error: the first ever run has no S3
cache, and we simply keep the committed one.

Usage (in CI, after `aws s3 cp ... /tmp/s3-cache.json`):
    uv run scripts/merge_cache.py \
        --committed ./data-cache.json \
        --s3 /tmp/s3-cache.json \
        --out ./data-cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, float]:
    with path.open("r") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committed", type=Path, default=Path("./data-cache.json"))
    parser.add_argument("--s3", type=Path, required=True,
                        help="S3-downloaded cache (may not exist on the first run)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: --committed, in place)")
    args = parser.parse_args(argv)
    out = args.out or args.committed

    committed = load(args.committed)

    if not args.s3.exists() or args.s3.stat().st_size == 0:
        print(f"No S3 cache at {args.s3}; keeping committed cache "
              f"({len(committed)} entries) unchanged.")
        if out != args.committed:
            with out.open("w") as f:
                json.dump(committed, f, sort_keys=True, indent=1)
        return 0

    s3 = load(args.s3)
    merged = dict(s3)
    merged.update(committed)  # committed wins on shared keys
    added = len(merged) - len(committed)
    print(
        f"Merged S3 cache: committed={len(committed)}, s3={len(s3)}, "
        f"merged={len(merged)} (+{added} recent keys from S3)"
    )
    with out.open("w") as f:
        json.dump(merged, f, sort_keys=True, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
