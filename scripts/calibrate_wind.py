"""
Check the acoustic wind detector against the station's anemometer.

`soundsanity.wind` ships a default threshold on `wind_index`. This script says
what that threshold is actually worth on this corpus: it sweeps candidate
thresholds, scores each against a wind speed the station measured, and reports
the one that separates best.

Two things it cannot do, and neither is a defect in the code. The station is
kilometres from most sites, so its wind is an index of the day's weather rather
than a reading at the microphone; and a recording made in a lull during a gale
is genuinely calm even when the station is not. Both put a ceiling on the
agreement well below 1.0. The sweep is still the right way to choose a
threshold — just read the resulting numbers as "as good as this station can
tell us", not as detector accuracy.

Usage:

    uv run python scripts/calibrate_wind.py
    uv run python scripts/calibrate_wind.py --windy-knots 25 --by site
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from soundsanity.meteo import join_meteo, load_meteo, verify_time_alignment

ANALYSIS = Path("data/analysis")
INVENTORY = Path("data/inventory.parquet")
METEO = Path("meteo-data/Meteo_2022-2025.xlsx")


def load_joined(analysis_dir, inventory_path, meteo_path, medium="air"):
    """
    Load the analysis shards, attach deployment metadata, and join weather.

    Returns:
        tuple of (pandas.DataFrame, pandas.DataFrame): The joined recordings and
        the weather table.

    Raises:
        FileNotFoundError: If no analysis shards exist yet.
    """
    shards = sorted(Path(analysis_dir).glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no analysis shards in {analysis_dir}; run scripts/analyze_corpus.py first"
        )

    analysis = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)
    inventory = pd.read_parquet(inventory_path)

    frame = analysis.merge(
        inventory.drop(columns=["file_name"]), on="path", how="left", suffixes=("", "_inv")
    )
    if medium:
        frame = frame[frame["medium"] == medium]

    meteo = load_meteo(meteo_path)
    return join_meteo(frame.dropna(subset=["timestamp_utc"]), meteo), meteo


def sweep_threshold(frame, windy_knots=20.0, metric="wind_index",
                    thresholds=np.arange(0.05, 1.0, 0.05), direction="above"):
    """
    Score candidate thresholds against a wind speed the station measured.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        windy_knots (float): Station wind at or above which a recording counts
            as truly windy.
        metric (str): The acoustic column to threshold.
        thresholds (iterable of float): Candidates to try.
        direction (str): ``"above"`` if larger values mean more wind, ``"below"``
            if smaller ones do. `spectral_tilt` falls with wind, and scoring it
            the wrong way round makes a genuinely informative metric look
            worthless.

    Returns:
        pandas.DataFrame: ``threshold``, ``precision``, ``recall``,
        ``specificity``, ``f1``, ``youden`` and ``flagged_rate`` per candidate.

    Raises:
        ValueError: If `direction` is not "above" or "below".
    """
    if direction not in ("above", "below"):
        raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")

    pair = frame[[metric, "wind_knots"]].dropna()
    truth = pair["wind_knots"] >= windy_knots

    rows = []
    for threshold in thresholds:
        predicted = (
            pair[metric] >= threshold if direction == "above"
            else pair[metric] <= threshold
        )
        true_pos = int((predicted & truth).sum())
        false_pos = int((predicted & ~truth).sum())
        false_neg = int((~predicted & truth).sum())
        true_neg = int((~predicted & ~truth).sum())

        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else np.nan
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else np.nan
        specificity = true_neg / (true_neg + false_pos) if true_neg + false_pos else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and not np.isnan(precision * recall)
            else np.nan
        )

        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "youden": recall + specificity - 1 if not np.isnan(specificity) else np.nan,
                "flagged_rate": float(predicted.mean()),
            }
        )

    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--meteo", type=Path, default=METEO)
    parser.add_argument("--windy-knots", type=float, default=20.0)
    parser.add_argument("--metric", default="wind_index")
    parser.add_argument("--direction", choices=["above", "below"], default="above",
                        help="'below' for metrics that fall as wind rises, "
                             "such as spectral_tilt.")
    parser.add_argument("--by", nargs="+", default=["site"])
    args = parser.parse_args(argv)

    frame, meteo = load_joined(args.analysis, args.inventory, args.meteo)
    print(f"{len(frame)} airborne recordings, "
          f"{frame['wind_knots'].notna().sum()} matched to weather\n")

    print("=== clock alignment (correlation of lf_level_db with wind, by shift) ===")
    align = verify_time_alignment(frame, meteo, metric="lf_level_db")
    for _, row in align.iterrows():
        marker = "  <-- assumed" if row["offset_h"] == 0 else ""
        print(f"  {row['offset_h']:+3.0f} h   r = {row['correlation']: .3f}   "
              f"n = {row['n']:6.0f}{marker}")
    best = align.loc[align["correlation"].idxmax()]
    verdict = "confirmed" if best["offset_h"] == 0 else "WRONG — see peak above"
    print(f"  best shift: {best['offset_h']:+.0f} h  ->  UTC-3 assumption {verdict}\n")

    print("=== correlation with measured wind ===")
    for column in ("wind_index", "lf_level_db", "lf_ratio", "rms_db",
                   "noise_floor_db", "spectral_tilt"):
        if column not in frame.columns:
            continue
        pair = frame[[column, "wind_knots"]].dropna()
        print(f"  {column:16} r = {pair[column].corr(pair['wind_knots']): .3f}")

    print(f"\n=== threshold sweep (truth: station wind >= {args.windy_knots:.0f} kt) ===")
    sweep = sweep_threshold(frame, args.windy_knots, args.metric,
                            direction=args.direction)
    print(sweep.round(3).to_string(index=False))

    best_row = sweep.loc[sweep["youden"].idxmax()]
    print(f"\n  best separation at {args.metric} >= {best_row['threshold']:.2f} "
          f"(Youden J = {best_row['youden']:.3f}, "
          f"precision {best_row['precision']:.2f}, recall {best_row['recall']:.2f})")

    print(f"\n=== per-{'/'.join(args.by)} correlation ===")
    for key, group in frame.groupby(args.by, observed=True):
        pair = group[[args.metric, "wind_knots"]].dropna()
        if len(pair) < 30:
            continue
        print(f"  {str(key):34} n={len(pair):6d}  "
              f"r={pair[args.metric].corr(pair['wind_knots']): .3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
