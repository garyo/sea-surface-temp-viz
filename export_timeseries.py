#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Export aggregated time-series from sst-data-cache.json to JSON for the web frontend.

Phase 1: only the global region from the OISST source. Future phases will add
more regions (Niño 3.4, basins, hemispheres) and more sources (ERA5, MODIS) by
extending the same schema.

Schema (one file per region):

    {
      "region": "global",
      "region_label": "Global (60°S–60°N)",
      "sources": {
        "oisst": {
          "datasets": {
            "sst":  { "dates": ["1982-01-01", ...], "values": [20.07, ...] },
            "anom": { "dates": [...],               "values": [...] }
          }
        }
      },
      "updated": "2026-05-08T13:15:00+00:00"
    }
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(sst|anom)$")

REGIONS: dict[str, dict[str, str]] = {
    "global": {"label": "Global (60°S–60°N)"},
}


def load_cache(path: Path) -> dict[str, float]:
    with path.open("r") as f:
        return json.load(f)


def split_by_dataset(cache: dict[str, float]) -> dict[str, list[tuple[str, float]]]:
    """Group cache entries by dataset name, sorted by date."""
    by_dataset: dict[str, list[tuple[str, float]]] = {"sst": [], "anom": []}
    for key, value in cache.items():
        m = KEY_RE.match(key)
        if not m:
            continue
        date, dataset = m.group(1), m.group(2)
        if isinstance(value, float) and math.isnan(value):
            continue
        by_dataset[dataset].append((date, value))
    for entries in by_dataset.values():
        entries.sort(key=lambda kv: kv[0])
    return by_dataset


def build_global_payload(cache: dict[str, float]) -> dict[str, Any]:
    by_dataset = split_by_dataset(cache)
    datasets: dict[str, dict[str, list]] = {}
    for name, entries in by_dataset.items():
        datasets[name] = {
            "dates": [d for d, _ in entries],
            "values": [v for _, v in entries],
        }
    return {
        "region": "global",
        "region_label": REGIONS["global"]["label"],
        "sources": {
            "oisst": {"datasets": datasets},
        },
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export aggregated time-series from sst-data-cache.json to JSON."
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("./sst-data-cache.json"),
        help="Input cache file produced by sea-surface-temps.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./output/timeseries"),
        help="Output directory; one JSON per region is written here",
    )
    args = parser.parse_args(argv)

    if not args.cache_file.exists():
        print(f"❌ Cache file not found: {args.cache_file}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_file)
    payload = build_global_payload(cache)

    out_path = args.out_dir / "global.json"
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))

    sst_n = len(payload["sources"]["oisst"]["datasets"]["sst"]["dates"])
    anom_n = len(payload["sources"]["oisst"]["datasets"]["anom"]["dates"])
    size_kb = out_path.stat().st_size / 1024
    print(f"✓ Wrote {out_path} ({size_kb:.1f} KB; sst={sst_n}, anom={anom_n})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
