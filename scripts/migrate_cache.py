#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# SPDX-License-Identifier: MIT
"""One-time migration: sst-data-cache.json → data-cache.json.

Old key shape: ``YYYY-MM-DD-{sst|anom}``
New key shape: ``YYYY-MM-DD-{source}-{dataset}-{region}``

All old entries are translated to source=oisst, region=global (the existing
pipeline produced 60°S–60°N global aggregates for SST and anomaly only).

Run once locally, then commit data-cache.json. The old file can stay for one
release as a safety net before being removed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OLD_KEY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(sst|anom)$")


def migrate(in_path: Path, out_path: Path) -> int:
    with in_path.open("r") as f:
        old: dict[str, float] = json.load(f)

    new: dict[str, float] = {}
    skipped: list[str] = []
    for key, value in old.items():
        m = OLD_KEY_RE.match(key)
        if not m:
            skipped.append(key)
            continue
        date, dataset = m.group(1), m.group(2)
        new_key = f"{date}-oisst-{dataset}-global"
        new[new_key] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(new, f, sort_keys=True, indent=2)

    print(f"Migrated {len(old)} → {len(new)} entries")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} unrecognized keys:")
        for k in skipped[:10]:
            print(f"     {k}")
        if len(skipped) > 10:
            print(f"     ... and {len(skipped) - 10} more")
        return 1
    print(f"✓ Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", type=Path, default=Path("./sst-data-cache.json"))
    parser.add_argument("--out", type=Path, default=Path("./data-cache.json"))
    args = parser.parse_args(argv)
    if not args.infile.exists():
        print(f"❌ Input cache not found: {args.infile}", file=sys.stderr)
        return 1
    return migrate(args.infile, args.out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
