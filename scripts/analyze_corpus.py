"""
Run the quality and wind analysis over the inventoried field corpus.

Takes `data/inventory.parquet` as its worklist and writes results as parquet
shards under `data/analysis/`. The run is resumable: every shard is written
before the next chunk starts, and on restart any file already present in a
shard is skipped. Interrupting it costs at most one chunk.

Usage:

    # everything, resuming whatever a previous run finished
    uv run python scripts/analyze_corpus.py

    # a stratified sample, for a first look
    uv run python scripts/analyze_corpus.py --per-site-season 800 --out data/analysis_pilot

    # one season only
    uv run python scripts/analyze_corpus.py --seasons 2024-25
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from soundsanity.field import DEFAULT_WINDOW_S, analyze_field_task

INVENTORY = Path("data/inventory.parquet")
OUT_DIR = Path("data/analysis")
CHUNK = 2000


def load_worklist(args):
    """
    Select the recordings this run should analyze.

    Returns:
        pandas.DataFrame: The inventory rows to process, already filtered and
        sampled but not yet checked against completed shards.
    """
    inventory = pd.read_parquet(args.inventory)

    # Files whose header could not be read have no reliable duration or
    # timestamp; analyzing them would produce rows that cannot be joined.
    usable = inventory[inventory["header_error"].fillna("") == ""].copy()
    usable = usable.dropna(subset=["timestamp_utc"])

    if args.seasons:
        usable = usable[usable["season"].isin(args.seasons)]
    if args.sites:
        usable = usable[usable["site"].isin(args.sites)]
    if args.medium:
        usable = usable[usable["medium"] == args.medium]

    # A recording shorter than the window cannot supply a comparable
    # measurement, and is nearly always a truncated file at the end of a
    # deployment.
    if args.window_s:
        usable = usable[usable["duration_s"].fillna(0) >= args.window_s * 0.99]

    if args.per_site_season:
        # Sample evenly in time rather than at random, so a subsample still
        # covers the whole deployment and the full range of weather it saw.
        usable = usable.sort_values("timestamp_utc")
        groups = [
            group.iloc[:: max(1, len(group) // args.per_site_season)]
            for _, group in usable.groupby(["season", "site"], observed=True)
        ]
        usable = pd.concat(groups, ignore_index=True) if groups else usable

    return usable.sort_values(["season", "site", "timestamp_utc"])


def completed_paths(out_dir):
    """Return the set of paths already recorded in shards under `out_dir`."""
    shards = sorted(out_dir.glob("part-*.parquet"))
    if not shards:
        return set(), 0
    done = set()
    for shard in shards:
        done.update(pd.read_parquet(shard, columns=["path"])["path"])
    return done, len(shards)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seasons", nargs="+")
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--medium", choices=["air", "underwater"])
    parser.add_argument("--per-site-season", type=int,
                        help="Analyze about this many files per site per season.")
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    work = load_worklist(args)
    args.out.mkdir(parents=True, exist_ok=True)
    done, shard_count = completed_paths(args.out)

    pending = work[~work["path"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)

    print(
        f"worklist {len(work)} files | already done {len(done)} | "
        f"to analyze {len(pending)}",
        file=sys.stderr,
    )
    if pending.empty:
        print("nothing to do", file=sys.stderr)
        return 0

    kwargs = {"window_s": args.window_s}
    paths = pending["path"].tolist()
    started = time.time()
    completed = 0

    for offset in range(0, len(paths), args.chunk):
        batch = paths[offset : offset + args.chunk]
        tasks = [(path, kwargs) for path in batch]

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(analyze_field_task, tasks, chunksize=8))

        shard = args.out / f"part-{shard_count:05d}.parquet"
        pd.DataFrame(rows).to_parquet(shard, index=False)
        shard_count += 1
        completed += len(batch)

        elapsed = time.time() - started
        rate = completed / elapsed
        remaining = (len(paths) - completed) / rate if rate else 0
        errors = sum(1 for row in rows if row["error"])
        print(
            f"  {completed}/{len(paths)} ({rate:.1f} files/s, "
            f"~{remaining / 60:.0f} min left, {errors} errors in chunk) -> {shard.name}",
            file=sys.stderr,
        )

    print(f"done in {(time.time() - started) / 60:.1f} min", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
