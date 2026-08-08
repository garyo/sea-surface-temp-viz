#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: MIT
"""Export aggregated time-series from data-cache.json.gz to one JSON per region.

Reads the on-disk cache populated by pipeline.py — keys are
``YYYY-MM-DD-{source}-{dataset}-{region}`` — and emits

    {out_dir}/{region}.json

for every region present in the cache, in this schema (parallel arrays, ~40%
smaller than array-of-objects, ECharts consumes natively):

    {
      "region": "nino_3_4",
      "region_label": "Niño 3.4 (5°S–5°N, 170°W–120°W)",
      "sources": {
        "oisst": {
          "datasets": {
            "sst":  { "dates": ["1982-01-01", ...], "values": [26.81, ...] },
            "anom": { "dates": [...],               "values": [...] }
          },
          "preliminary_from": "2026-08-01"
        }
      },
      "updated": "..."
    }

``preliminary_from`` is optional and present only when the source's most recent
days are still provisional (NOAA publishes OISST ``_preliminary`` files for ~2
weeks before revising them). Every date at or after it is subject to revision;
the frontend draws that tail as a dotted line. It is derived from the
``{date}-{source}-preliminary-flag`` cache keys written by pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import cache_io
import regions

# Matches keys like 1982-01-01-oisst-sst-nino_3_4 or 2026-01-15-era5-sst_anom-n_hemi.
# Dataset and region can both contain underscores; source cannot (so the
# greedy match for source still stops at the first `-`).
KEY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-([a-z0-9]+)-([a-z0-9_]+)-([a-z0-9_]+)$")

# Reserved (dataset, region) pair that pipeline.py's preliminary_key() uses to
# mark a date whose source file was provisional. Neither is a real dataset or
# region, so these keys are pulled out before grouping.
PRELIM_DATASET = "preliminary"
PRELIM_REGION = "flag"

# Only trust a preliminary flag this close to the series' latest date. The daily
# workflow prunes and re-fetches the last 90 days, so any flag inside that window
# reflects the current upstream state; an older one is a stale leftover and would
# otherwise dot months of settled data.
PRELIM_MAX_AGE_DAYS = 90


load_cache = cache_io.load_cache


def group_cache(
    cache: dict[str, float],
) -> tuple[
    dict[str, dict[str, dict[str, list[tuple[str, float]]]]],
    dict[str, set[str]],
]:
    """Group entries by region → source → dataset → [(date, value), ...].

    Also returns the preliminary flags as source → {date, ...}.
    """
    grouped: dict[str, dict[str, dict[str, list[tuple[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    prelim: dict[str, set[str]] = defaultdict(set)
    skipped = 0
    for key, value in cache.items():
        m = KEY_RE.match(key)
        if not m:
            skipped += 1
            continue
        date, source, dataset, region = m.groups()
        if dataset == PRELIM_DATASET and region == PRELIM_REGION:
            prelim[source].add(date)
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        grouped[region][source][dataset].append((date, value))
    if skipped:
        print(f"⚠️  Skipped {skipped} cache entries that didn't match the key regex")
    # Sort each list by date
    for r in grouped.values():
        for s in r.values():
            for entries in s.values():
                entries.sort(key=lambda kv: kv[0])
    return grouped, prelim


def preliminary_from(
    datasets: dict[str, list[tuple[str, float]]],
    flagged: set[str],
) -> str | None:
    """Earliest still-provisional date for one source, or None.

    Preliminary days are always a trailing run, so one date is enough to describe
    them — far cheaper than a boolean parallel to every value.
    """
    latest = max(
        (entries[-1][0] for entries in datasets.values() if entries), default=None
    )
    if latest is None:
        return None
    cutoff = (
        date.fromisoformat(latest) - timedelta(days=PRELIM_MAX_AGE_DAYS)
    ).isoformat()
    recent = [d for d in flagged if d >= cutoff]
    return min(recent) if recent else None


def build_payload(
    region_id: str,
    sources: dict[str, dict[str, list[tuple[str, float]]]],
    prelim: dict[str, set[str]],
) -> dict[str, Any]:
    sources_out: dict[str, dict[str, Any]] = {}
    for source_id, datasets in sources.items():
        ds_out: dict[str, dict[str, list]] = {}
        for ds_name, entries in datasets.items():
            ds_out[ds_name] = {
                "dates": [d for d, _ in entries],
                "values": [v for _, v in entries],
            }
        sources_out[source_id] = {"datasets": ds_out}
        first_prelim = preliminary_from(datasets, prelim.get(source_id, set()))
        if first_prelim:
            sources_out[source_id]["preliminary_from"] = first_prelim
    label = regions.label_for(region_id) if region_id in regions.REGIONS else region_id
    return {
        "region": region_id,
        "region_label": label,
        "sources": sources_out,
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def report_recent_gaps(
    grouped: dict[str, dict[str, dict[str, list[tuple[str, float]]]]],
    days: int,
) -> int:
    """Print interior gaps in the recent tail of each region/source/dataset.

    An "interior gap" is a calendar day missing *before* the latest day we have
    (days after the latest are just not-yet-published, e.g. ERA5T latency, and
    are expected). Surfacing these in the CI log turns a silent upstream outage
    into something visible. Returns the total number of (series, missing-day)
    pairs found.
    """
    total = 0
    for region_id in sorted(grouped):
        for source_id, datasets in sorted(grouped[region_id].items()):
            for ds_name, entries in sorted(datasets.items()):
                if not entries:
                    continue
                have = {d for d, _ in entries}
                latest = date.fromisoformat(max(have))
                start = latest - timedelta(days=days)
                missing = [
                    (start + timedelta(days=i)).isoformat()
                    for i in range((latest - start).days + 1)
                    if (start + timedelta(days=i)).isoformat() not in have
                ]
                if missing:
                    total += len(missing)
                    shown = ", ".join(missing[:8])
                    more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
                    print(
                        f"⚠️  gap: {region_id}/{source_id}/{ds_name} missing "
                        f"{len(missing)} day(s) in last {days}d: {shown}{more}"
                    )
    if total:
        print(
            f"⚠️  {total} missing recent day(s) across all series — the daily "
            f"backfill will retry these until the upstream source republishes."
        )
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("./data-cache.json.gz"),
        help="Input cache file produced by pipeline.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./output/timeseries"),
        help="Output directory; one JSON per region is written here",
    )
    parser.add_argument(
        "--min-dates",
        type=int,
        default=365,
        help="Skip regions with fewer than this many dated values "
        "(prevents thin-data uploads before the local backfill is done)",
    )
    parser.add_argument(
        "--gap-report-days",
        type=int,
        default=120,
        help="Report interior missing days within this many days of each "
        "series' latest date (visibility for upstream outages)",
    )
    args = parser.parse_args(argv)

    if not args.cache_file.exists():
        print(f"❌ Cache file not found: {args.cache_file}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_file)
    grouped, prelim = group_cache(cache)

    if not grouped:
        print("⚠️  No regions found in cache; nothing to write")
        return 0

    # Only emit a region file once it has enough history for the year-overlay
    # chart to look meaningful. Without the local NetCDF backfill, the daily
    # cron will only have ~90 days for non-global regions — emitting that
    # thin slice would put a sparse selector on the live site.
    MIN_DATES = args.min_dates

    for region_id, sources in sorted(grouped.items()):
        max_dates = max(
            (
                len(entries)
                for datasets in sources.values()
                for entries in datasets.values()
            ),
            default=0,
        )
        if max_dates < MIN_DATES:
            print(
                f"⏭  {region_id}: only {max_dates} dates "
                f"(< {MIN_DATES}); skipping. Run the backfill to populate."
            )
            continue

        payload = build_payload(region_id, sources, prelim)
        out_path = args.out_dir / f"{region_id}.json"
        with out_path.open("w") as f:
            json.dump(payload, f, separators=(",", ":"))
        ds_summary = ", ".join(
            f"{src}.{ds}={len(entries)}"
            for src, datasets in sources.items()
            for ds, entries in sorted(datasets.items())
        )
        size_kb = out_path.stat().st_size / 1024
        print(f"✓ {out_path} ({size_kb:.1f} KB; {ds_summary})")

    report_recent_gaps(grouped, args.gap_report_days)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
