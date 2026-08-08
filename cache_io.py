# SPDX-License-Identifier: MIT
"""Read/write the regional-aggregate cache, transparently gzipped.

Every tool that touches ``data-cache.json.gz`` goes through here, so the on-disk
format is decided in exactly one place. A ``.gz`` suffix selects compression;
any other suffix is plain JSON, which keeps the older uncompressed files (and
the S3 object during migration) readable without a special case at the call
site.

Two properties worth keeping:

*Compression.* The cache is ~1.1M keys of `"YYYY-MM-DD-source-dataset-region":
float`, which is enormously repetitive and compresses ~8x. That matters because
the file is committed — at 58 MB plain it drew GitHub's large-file warning and
was heading for the 100 MB hard limit.

*Precision.* Values are area-weighted means of float32 grids; OISST's own data
is `int16 * 0.01`, so it carries two decimals of real signal. Writing 17
significant digits stored noise and cost ~20% of the file. Six decimals is
still far finer than anything physical here.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO, Any, cast

# Decimals kept per value. Well beyond the sources' own resolution; see above.
PRECISION = 6


def _open(path: Path, mode: str, *, compress: bool) -> IO[str]:
    # `compress` is passed explicitly rather than re-derived from `path`: the
    # atomic write below goes to a `.part` temp file, whose own suffix says
    # nothing about the format the final name calls for.
    # (`mode` is dynamic, so the stdlib overloads can't resolve statically;
    # both branches do return text handles.)
    if compress:
        return cast(
            IO[str], gzip.open(path, mode + "t", encoding="utf-8", compresslevel=9)
        )
    return cast(IO[str], path.open(mode, encoding="utf-8"))


def is_compressed(path: Path) -> bool:
    return Path(path).suffix == ".gz"


def load_cache(path: Path) -> dict[str, float]:
    """Load a cache file. Raises OSError if it doesn't exist."""
    path = Path(path)
    with _open(path, "r", compress=is_compressed(path)) as f:
        return json.load(f)


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write a cache file atomically.

    Via a temp file + rename because pipeline.py re-saves after every fetched
    date; a crash partway through a direct write would leave a truncated cache
    that no longer parses.
    """
    path = Path(path)
    compact = {k: round(float(v), PRECISION) for k, v in cache.items()}
    tmp = path.with_name(path.name + ".part")
    with _open(tmp, "w", compress=is_compressed(path)) as f:
        json.dump(compact, f, sort_keys=True, indent=2)
    tmp.replace(path)
