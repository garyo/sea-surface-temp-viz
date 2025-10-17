#!/usr/bin/env bash

# Number of parallel jobs (override with: N=16 ./regen-maps.sh)
N=${N:-8}

# Convert date (YYYY-MM-DD) or days-ago number to days-ago
# If input looks like a date, compute days ago from today
# Otherwise, treat as days-ago directly
to_days_ago() {
    local input=$1
    # Check if input looks like a date (contains dashes and matches pattern)
    if [[ $input =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        local today_epoch=$(date +%s)
        # macOS date command syntax
        if [[ "$OSTYPE" == "darwin"* ]]; then
            local date_epoch=$(date -j -f "%Y-%m-%d" "$input" +%s)
        else
            # Linux/GNU date syntax
            local date_epoch=$(date -d "$input" +%s)
        fi
        local diff_seconds=$((today_epoch - date_epoch))
        local days_ago=$((diff_seconds / 86400))
        echo $days_ago
    else
        echo $input
    fi
}

# Start and end can be dates (YYYY-MM-DD) or days-ago (default 0 to 360)
START_DAY=$(to_days_ago "${1:-0}")
END_DAY=$(to_days_ago "${2:-360}")

echo '# Saving maps into maps/ folder'
echo "Using $N parallel processes"
echo "Processing days $START_DAY to $END_DAY"

# Generate all combinations and run in parallel
parallel -j $N --line-buffer \
  'echo "Doing {1} days ago, dataset {2}"; uv run sea-surface-temps.py --mode texture --dataset {2} --days-ago {1} --out maps' \
  ::: $(seq $START_DAY $END_DAY) ::: anom sst
