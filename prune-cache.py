# SPDX-License-Identifier: MIT

import sys
import json
import datetime
import argparse
import pathlib

def parse_cache_date(datestr: str):
    yr, mo, day, type = datestr.split('-')
    date = datetime.date(int(yr), int(mo), int(day))
    return date

def prune_cache_file(infile: str, outfile: str, days_before_today):
    with open(infile, 'r') as f:
        json_data = json.load(f)

    pruned = {}
    today = datetime.date.today()
    for key,value in json_data.items():
        date = parse_cache_date(key)
        delta = (today - date).days
        # print(f"{today} - {date}: delta={delta}")
        if delta > days_before_today:
            pruned[key] = value

    with open(outfile, 'w') as dstf:
        json.dump(pruned, dstf)

def main(argv=None):
    temps_cache_file='./sst-data-cache.json'
    pruned_file='./sst-data-cache-pruned.json'

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass

    try:
        parser = argparse.ArgumentParser(description="""Cache-file pruner""",
                                         formatter_class=CustomFormatter)
        parser.add_argument('--verbose', '-v', action='store_true',
                            help="""Process verbosely""")
        parser.add_argument('--days', '-d', type=int,
                            default=180,
                            help="""Prune limit: days before today -- most recent N days will be pruned out""")
        parser.add_argument('--inplace', '-I', action='store_true',
                            default = False,
                            help="""Write pruned result to input file, replacing it (out file param will be ignored)""")
        parser.add_argument('--out', '-o', type=pathlib.Path,
                            default = pruned_file,
                            help="""Write pruned result to this file""")
        parser.add_argument('--in', '-i', type=pathlib.Path,
                            dest='infile', # needed since "in" is a keyword
                            default='./sst-data-cache.json',
                            help="""Input cache file""")
        args = parser.parse_args(argv)

        if args.inplace:
            args.out = args.infile
        prune_cache_file(args.infile, args.out, args.days)

    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
