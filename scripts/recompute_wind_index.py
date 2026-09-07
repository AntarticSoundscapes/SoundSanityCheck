"""
Reapply the wind index to an existing analysis after a recalibration.

`wind_index` and `is_windy` are pure functions of `lf_ratio` and `lf_level_db`,
both of which every analysis shard already stores. Refitting the thresholds
therefore does not require decoding 26,000 recordings again — the two columns
can simply be recomputed in place.

Usage:

    uv run python scripts/recompute_wind_index.py
    uv run python scripts/recompute_wind_index.py --min-wind-index 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soundsanity.wind import WIND_CONFIG, wind_index_from_features

ANALYSIS = Path("data/analysis")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--min-wind-index", type=float)
    parser.add_argument("--wind-lf-level-db", type=float)
    parser.add_argument("--wind-level-span-db", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    overrides = {
        key: value
        for key, value in (
            ("min_wind_index", args.min_wind_index),
            ("wind_lf_level_db", args.wind_lf_level_db),
            ("wind_level_span_db", args.wind_level_span_db),
        )
        if value is not None
    }
    config = {**WIND_CONFIG, **overrides}
    print("applying: " + ", ".join(
        f"{k}={config[k]}"
        for k in ("min_wind_index", "wind_lf_level_db",
                  "wind_level_span_db", "lf_ratio_floor")
    ))

    shards = sorted(args.analysis.glob("part-*.parquet"))
    if not shards:
        print(f"no shards in {args.analysis}")
        return 1

    before = after = total = 0
    for shard in shards:
        frame = pd.read_parquet(shard)
        index, windy = wind_index_from_features(
            frame["lf_ratio"], frame["lf_level_db"], config
        )
        before += int(frame["is_windy"].fillna(False).sum())
        after += int(windy.sum())
        total += len(frame)

        if not args.dry_run:
            frame["wind_index"] = index
            frame["is_windy"] = windy
            frame.to_parquet(shard, index=False)

    verb = "would flag" if args.dry_run else "flags"
    print(f"{len(shards)} shards, {total:,} recordings")
    print(f"  was:  {before:,} windy ({before / total:.1%})")
    print(f"  {verb}: {after:,} windy ({after / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
