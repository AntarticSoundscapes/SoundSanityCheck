"""
Inventory the field corpus on the external drive.

Walks each season folder, reads every recording's header, and writes one
parquet file describing the whole corpus: when and where each file was
recorded, and at what gain. Nothing here decodes audio — the run is bounded by
how fast the drive answers seeks, not by CPU.

Run this before `analyze_corpus.py`; that script takes the inventory as its
worklist.

Usage:

    uv run python scripts/build_inventory.py
    uv run python scripts/build_inventory.py --seasons CAV_2024-2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from soundsanity.corpus import INVENTORY_FIELDS, inventory

DRIVE = Path("/Volumes/LaCie")
SEASONS = ("CAV_2022-2023", "CAV_2023-2024", "CAV_2024-2025")
OUT = Path("data/inventory.parquet")


def _progress(label, every=500):
    """Return a progress callback that rewrites a single terminal line."""
    started = time.time()

    def report(done, total):
        if done % every and done != total:
            return
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        sys.stderr.write(
            f"\r  {label}: {done}/{total} "
            f"({rate:.0f} files/s, ~{remaining / 60:.1f} min left)   "
        )
        sys.stderr.flush()

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=Path, default=DRIVE)
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    frames = []
    for season in args.seasons:
        root = args.drive / season
        if not root.is_dir():
            print(f"skipping {season}: not found under {args.drive}", file=sys.stderr)
            continue
        print(f"scanning {season} ...", file=sys.stderr)
        rows = inventory(root, jobs=args.jobs, progress=_progress(season))
        print(f"\n  {season}: {len(rows)} files", file=sys.stderr)
        frames.append(pd.DataFrame(rows, columns=list(INVENTORY_FIELDS)))

    if not frames:
        print("no seasons found", file=sys.stderr)
        return 1

    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"\nwrote {len(table)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
