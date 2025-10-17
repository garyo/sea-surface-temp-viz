#!/usr/bin/env bash

# Number of parallel jobs (override with: N=16 ./regen-maps.sh)
N=${N:-8}

# Start and end days-ago (default 0 to 360)
START_DAY=${1:-0}
END_DAY=${2:-360}

echo '# Saving maps into maps/ folder'
echo "Using $N parallel processes"
echo "Processing days $START_DAY to $END_DAY"

# Generate all combinations and run in parallel
parallel -j $N --line-buffer \
  'echo "Doing {1} days ago, dataset {2}"; uv run sea-surface-temps.py --mode texture --dataset {2} --days-ago {1} --out maps' \
  ::: $(seq $START_DAY $END_DAY) ::: anom sst
