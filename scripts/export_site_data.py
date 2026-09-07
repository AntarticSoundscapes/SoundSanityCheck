"""
Aggregate the analysis shards into the JSON the showcase site reads.

The site is a static page: it ships the numbers, not the corpus. Everything
here collapses 26k per-recording rows into a few hundred site x season x band
cells, which is small enough to embed directly in the HTML and keeps the
result a single file with no server behind it.

Usage:

    uv run python scripts/export_site_data.py
    uv run python scripts/export_site_data.py --out site/data.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from soundsanity.meteo import WIND_BAND_LABELS, WIND_SCALE_BREAK
from soundsanity.report import (
    apply_gain_correction,
    deployment_quality,
    direction_response,
    standardize_by_wind,
    wind_response,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_wind import ANALYSIS, INVENTORY, METEO, load_joined  # noqa: E402

METRIC = "lf_level_db_gc"
SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def clean(value):
    """Return a JSON-safe number, mapping NaN and infinities to None."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else round(number, 3)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def records(frame):
    """Convert a DataFrame to a list of JSON-safe dicts."""
    return [
        {key: clean(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def build(frame, inventory):
    """
    Build the full payload for the site.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather, gain-corrected.
        inventory (pandas.DataFrame): The full field inventory, used for the
            reliability denominator (the analysis worklist already excludes
            unreadable files, so error rates must come from the inventory).

    Returns:
        dict: The payload, ready to serialize.
    """
    payload = {}

    # --- Sites -----------------------------------------------------------
    seasons = sorted(frame["season"].dropna().unique())
    site_rows = []
    for (site,), group in frame.groupby(["site"], observed=True):
        site_seasons = sorted(group["season"].dropna().unique())
        site_rows.append(
            {
                "site": site,
                "medium": group["medium"].mode().iloc[0],
                "seasons": site_seasons,
                "n": int(len(group)),
                "spans_break": len(set(site_seasons) & {"2023-24", "2024-25"}) == 2,
            }
        )
    payload["sites"] = sorted(site_rows, key=lambda row: -row["n"])
    payload["seasons"] = list(seasons)
    payload["wind_bands"] = list(WIND_BAND_LABELS)

    # --- Dose-response: site x season x wind band -------------------------
    dose = wind_response(frame, metric=METRIC, by=("site", "season"), min_count=8)
    sat = (
        frame.dropna(subset=[METRIC, "wind_band"])
        .groupby(["site", "season", "wind_band"], observed=True)["is_saturated"]
        .mean()
        .rename("sat_rate")
        .reset_index()
    )
    dose = dose.merge(sat, on=["site", "season", "wind_band"], how="left")
    payload["dose"] = records(dose)

    # Pooled across seasons, per site: the comparison view puts several sites
    # on one chart, and splitting each of those by season too would be
    # unreadable. Seasons are safe to pool here — each row keeps its own
    # timestamped wind reading, so the wind axis stays valid even where the
    # 2022-23 scale break makes cross-season *comparison* unsafe elsewhere.
    dose_site = wind_response(frame, metric=METRIC, by=("site",), min_count=8)
    sat_site = (
        frame.dropna(subset=[METRIC, "wind_band"])
        .groupby(["site", "wind_band"], observed=True)["is_saturated"]
        .mean()
        .rename("sat_rate")
        .reset_index()
    )
    payload["dose_by_site"] = records(
        dose_site.merge(sat_site, on=["site", "wind_band"], how="left")
    )

    # Pooled across sites, for the overview curve.
    pooled = wind_response(frame, metric=METRIC, by=("season",), min_count=8)
    pooled_sat = (
        frame.dropna(subset=[METRIC, "wind_band"])
        .groupby(["season", "wind_band"], observed=True)["is_saturated"]
        .mean()
        .rename("sat_rate")
        .reset_index()
    )
    payload["dose_pooled"] = records(
        pooled.merge(pooled_sat, on=["season", "wind_band"], how="left")
    )

    # --- Wind rose: site x sector ----------------------------------------
    rose = direction_response(frame, metric=METRIC, by=("site",), sectors=8,
                              min_count=5, min_knots=10.0)
    rose["sector"] = (rose["sector_deg"] / 45).astype(int).map(
        dict(enumerate(SECTOR_NAMES))
    )
    payload["rose"] = records(rose)

    spread = (
        rose.groupby("site")
        .agg(spread=("mean", lambda s: s.max() - s.min()),
             worst=("mean", "idxmax"), sectors=("mean", "size"))
    )
    spread["worst_sector"] = rose.loc[spread["worst"], "sector"].to_numpy()
    spread["worst_db"] = rose.loc[spread["worst"], "mean"].to_numpy()
    payload["spread"] = records(
        spread.drop(columns=["worst"]).reset_index().sort_values(
            "spread", ascending=False
        )
    )

    # --- Did we learn: matched-wind comparison ---------------------------
    # Only across the shared wind scale; 2022-23 sits on the other side of the
    # instrument break and cannot share this axis.
    after = frame[frame["timestamp_utc"] >= WIND_SCALE_BREAK]
    learn = standardize_by_wind(after, metric=METRIC, by=("site", "season"))
    payload["learn"] = records(learn)

    # --- Reliability, from the full inventory ----------------------------
    # Same definition as notebook section 7, so the two never disagree:
    # `header_error` is an empty string when the header parsed, not NaN, and a
    # file that opens but holds no audio is just as lost as one that does not.
    inv = inventory.copy()
    inv["failed"] = inv["header_error"].fillna("").ne("") | inv[
        "duration_s"
    ].fillna(-1).eq(0)
    reliability = (
        inv.groupby(["season"], observed=True)
        .agg(n=("path", "size"), failed=("failed", "sum"),
             error_rate=("failed", "mean"))
        .reset_index()
    )
    payload["reliability"] = records(reliability)
    failures = (
        inv[inv["failed"]]
        .groupby(["season", "site"], observed=True)
        .size()
        .rename("failed")
        .reset_index()
        .sort_values("failed", ascending=False)
    )
    payload["failures"] = records(failures)

    quality = deployment_quality(frame, by=("season", "site"))
    payload["quality"] = records(quality)

    # --- Weather context: the scale break --------------------------------
    monthly = frame.copy()
    monthly["month"] = monthly["timestamp_utc"].dt.strftime("%Y-%m")
    month_wind = (
        monthly.groupby("month")
        .agg(n=("wind_knots", "size"), wind=("wind_knots", "mean"),
             wind_p90=("wind_knots", lambda s: s.quantile(0.9)))
        .reset_index()
    )
    payload["monthly"] = records(month_wind[month_wind["n"] >= 100])

    # --- Headline numbers -------------------------------------------------
    valid = frame.dropna(subset=[METRIC, "wind_knots"])
    snr_valid = frame.dropna(subset=["snr_db", "wind_knots"])
    # snr_db = active_level_db - noise_floor_db (the p90-p10 spread of frame
    # RMS). Wind is a continuous broadband noise, not a quiet backdrop with
    # occasional loud events, so it lifts both percentiles together (r=0.84
    # between them here). Below ~20 kt the gap widens as level rises faster
    # than the floor recovers; above ~20 kt the active level hits the
    # clipping ceiling while the floor keeps climbing, so the gap collapses
    # again. The two slopes have opposite sign and a flat linear correlation
    # is their sum, not evidence the metric is blind to wind - it is real,
    # just non-monotonic, which a single Pearson r cannot show.
    snr_low = snr_valid[snr_valid["wind_knots"] < 20]
    snr_high = snr_valid[snr_valid["wind_knots"] >= 20]
    payload["summary"] = {
        "n_analyzed": int(len(frame)),
        "n_inventory": int(len(inventory)),
        "n_inventory_air": int((inventory["medium"] == "air").sum()),
        "n_sites": int(frame["site"].nunique()),
        "correlation": clean(valid[METRIC].corr(valid["wind_knots"])),
        "correlation_snr": clean(snr_valid["snr_db"].corr(snr_valid["wind_knots"])),
        "correlation_snr_low": clean(snr_low["snr_db"].corr(snr_low["wind_knots"])),
        "correlation_snr_high": clean(snr_high["snr_db"].corr(snr_high["wind_knots"])),
        "saturated_rate": clean(frame["is_saturated"].mean()),
        "scale_break": str(WIND_SCALE_BREAK.date()),
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site/data.json", type=Path)
    args = parser.parse_args()

    frame, _ = load_joined(ANALYSIS, INVENTORY, METEO, medium="air")
    frame = apply_gain_correction(frame)
    inventory = pd.read_parquet(INVENTORY)

    payload = build(frame, inventory)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=None, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} rows")


if __name__ == "__main__":
    main()
