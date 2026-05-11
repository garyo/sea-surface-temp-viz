# SPDX-License-Identifier: MIT

import sys
import json
import datetime
import argparse
import pathlib


def parse_cache_date(datestr: str):
    # Keys look like "YYYY-MM-DD-source-dataset-region" (e.g.
    # "1982-01-01-oisst-sst-global"). Only the date prefix matters here.
    parts = datestr.split("-")
    yr, mo, day = parts[0], parts[1], parts[2]
    return datetime.date(int(yr), int(mo), int(day))


def cache_source(key: str) -> str | None:
    """Return the source segment of a cache key (e.g. "oisst", "era5") or None."""
    parts = key.split("-")
    return parts[3] if len(parts) >= 6 else None


def prune_cache_file(infile: str, outfile: str, days_before_today, sources):
    """Prune recent entries (last `days_before_today` days) for the named sources.

    Entries from sources NOT in `sources` are kept regardless of date — the
    daily workflow only re-fetches the listed sources, so pruning the others
    just creates gaps that never get filled back in.
    """
    with open(infile, "r") as f:
        json_data = json.load(f)

    pruned = {}
    today = datetime.date.today()
    for key, value in json_data.items():
        src = cache_source(key)
        if src not in sources:
            # Source not slated for re-fetch — keep regardless of age.
            pruned[key] = value
            continue
        date = parse_cache_date(key)
        delta = (today - date).days
        if delta > days_before_today:
            pruned[key] = value

    with open(outfile, "w") as dstf:
        json.dump(pruned, dstf, indent=1, sort_keys=True)


def main(argv=None):
    pruned_file = "./sst-data-cache-pruned.json"

    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    try:
        parser = argparse.ArgumentParser(
            description="""Cache-file pruner""",
            formatter_class=CustomFormatter,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="""Process verbosely""",
        )
        parser.add_argument(
            "--days",
            "-d",
            type=int,
            default=180,
            help="""Prune limit: days before today -- """
            """most recent N days will be pruned out""",
        )
        parser.add_argument(
            "--inplace",
            "-I",
            action="store_true",
            default=False,
            help="""Write pruned result to input file, """
            """replacing it (out file param will be ignored)""",
        )
        parser.add_argument(
            "--out",
            "-o",
            type=pathlib.Path,
            default=pruned_file,
            help="""Write pruned result to this file""",
        )
        parser.add_argument(
            "--in",
            "-i",
            type=pathlib.Path,
            dest="infile",  # needed since "in" is a keyword
            default="./data-cache.json",
            help="""Input cache file""",
        )
        parser.add_argument(
            "--sources",
            default="oisst",
            help="Comma-separated source ids to prune (default: oisst). "
                 "Sources not listed here are kept regardless of date — useful "
                 "when only some sources are re-fetched by the daily cron.",
        )
        args = parser.parse_args(argv)

        if args.inplace:
            args.out = args.infile
        sources = {s.strip() for s in args.sources.split(",") if s.strip()}
        prune_cache_file(args.infile, args.out, args.days, sources)

    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
