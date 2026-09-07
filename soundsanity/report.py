"""
Aggregations over recordings joined to weather.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corpus import GAIN_OFFSET_DB
from .meteo import WIND_BAND_LABELS

#: Level columns that `apply_gain_correction` will normalize if present.
LEVEL_COLUMNS = (
    "rms_db",
    "noise_floor_db",
    "active_level_db",
    "lf_level_db",
    "hf_level_db",
)


def apply_gain_correction(frame, columns=LEVEL_COLUMNS, suffix="_gc"):
    """
    Add gain-normalized copies of the level columns.

    Levels are referred back to the `medium` gain setting by subtracting each
    recording's nominal offset, so that a season recorded hotter does not read
    as a louder soundscape.

    Args:
        frame (pandas.DataFrame): Rows carrying a ``gain`` column.
        columns (iterable of str): Level columns to normalize. Missing ones are
            skipped.
        suffix (str): Appended to each new column's name.

    Returns:
        pandas.DataFrame: A copy with the corrected columns added, plus
        ``gain_offset_db``. Rows whose gain is unknown get NaN rather than an
        uncorrected value, so they drop out of comparisons instead of biasing
        them.

    Raises:
        ValueError: If `frame` has no ``gain`` column.
    """
    if "gain" not in frame.columns:
        raise ValueError("frame has no 'gain' column; run the inventory join first")

    out = frame.copy()
    offset = out["gain"].map(GAIN_OFFSET_DB)
    out["gain_offset_db"] = offset

    for column in columns:
        if column in out.columns:
            out[f"{column}{suffix}"] = out[column] - offset

    return out


def wind_response(frame, metric="lf_level_db_gc", by=("site",),
                  band_column="wind_band", min_count=5):
    """
    Summarize an acoustic metric against measured wind strength.

    This is the dose-response curve: how much louder the recording gets for a
    given increase in wind. Reported per group, because a sheltered site and an
    exposed one answer very differently.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        metric (str): The acoustic column to summarize.
        by (sequence of str): Grouping columns, e.g. ``("site", "season")``.
        band_column (str): The wind band column.
        min_count (int): Bands with fewer recordings than this are dropped,
            since a mean over two files says nothing.

    Returns:
        pandas.DataFrame: ``*by``, ``wind_band``, ``n``, ``mean``, ``median``,
        ``p90`` and ``windy_rate`` (the share flagged `is_windy`, if present).

    Raises:
        ValueError: If `metric` or `band_column` is missing.
    """
    for column in (metric, band_column):
        if column not in frame.columns:
            raise ValueError(f"frame has no column {column!r}")

    keys = list(by) + [band_column]
    aggregations = {
        "n": (metric, "size"),
        "mean": (metric, "mean"),
        "median": (metric, "median"),
        "p90": (metric, lambda s: s.quantile(0.9)),
    }
    if "is_windy" in frame.columns:
        aggregations["windy_rate"] = ("is_windy", "mean")

    table = (
        frame.dropna(subset=[metric, band_column])
        .groupby(keys, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    return table[table["n"] >= min_count].reset_index(drop=True)


def wind_weights(frame, band_column="wind_band"):
    """
    Return the pooled wind-band distribution to standardize against.

    Using the corpus's own pooled distribution — rather than, say, a flat one —
    keeps the standardized figures close to the raw ones and avoids giving a
    rare 40-knot band the same weight as the common 10-15 knot one.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        band_column (str): The wind band column.

    Returns:
        pandas.Series: Weights summing to 1, indexed by wind band.
    """
    counts = frame[band_column].value_counts(dropna=True)
    return (counts / counts.sum()).sort_index()


def standardize_by_wind(frame, metric="lf_level_db_gc", by=("season",),
                        band_column="wind_band", weights=None, min_count=5):
    """
    Compare groups as if each had met the same weather.

    Each group's mean is computed within wind band and then re-averaged using
    one shared set of band weights. Two seasons that differ only in how windy
    they were come out equal; two that differ in how the recorders were placed
    do not.

    A group is only scored on the bands it actually has data in, and the weights
    are renormalized over those bands. `coverage` reports how much of the
    reference distribution that accounts for — a group with low coverage is
    being compared on a narrow slice of conditions and should be read with care.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        metric (str): The acoustic column to standardize.
        by (sequence of str): Grouping columns, e.g. ``("season",)``.
        band_column (str): The wind band column.
        weights (pandas.Series, optional): Band weights. Defaults to the pooled
            distribution of `frame`.
        min_count (int): Minimum recordings per band for that band to count.

    Returns:
        pandas.DataFrame: ``*by``, ``n``, ``raw_mean``, ``standardized`` and
        ``coverage``, sorted by the grouping columns. `raw_mean` is the
        uncorrected average, kept alongside so the size of the weather's
        contribution is visible.
    """
    weights = wind_weights(frame, band_column) if weights is None else weights
    cells = wind_response(
        frame, metric=metric, by=by, band_column=band_column, min_count=min_count
    )

    rows = []
    for key, group in cells.groupby(list(by), observed=True):
        key = key if isinstance(key, tuple) else (key,)
        band_weights = weights.reindex(group[band_column]).to_numpy(dtype=float)
        coverage = np.nansum(band_weights)

        if not coverage > 0:
            standardized = np.nan
        else:
            standardized = float(
                np.nansum(group["mean"].to_numpy() * band_weights) / coverage
            )

        raw = frame
        for column, value in zip(by, key):
            raw = raw[raw[column] == value]

        rows.append(
            dict(
                zip(by, key),
                n=int(group["n"].sum()),
                raw_mean=float(raw[metric].mean()),
                standardized=standardized,
                coverage=float(coverage),
            )
        )

    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


def deployment_quality(frame, by=("season", "site")):
    """
    Summarize how well the recorders themselves performed.

    Separate from the wind question: saturation is a gain choice, silence and
    errors are hardware or card failures. These are the parts of "did we get
    better" that have nothing to do with where the microphone was pointed.

    Args:
        frame (pandas.DataFrame): Analyzed recordings, joined or not.
        by (sequence of str): Grouping columns.

    Returns:
        pandas.DataFrame: ``*by``, ``n``, and the share of recordings that were
        saturated, silent, clicky, windy and errored.
    """
    out = frame.copy()
    out["is_error"] = out["status"].eq("ERROR")

    flags = {
        "saturated_rate": "is_saturated",
        "silent_rate": "is_silent",
        "clicky_rate": "is_clicky",
        "windy_rate": "is_windy",
        "error_rate": "is_error",
    }
    aggregations = {"n": ("status", "size")}
    for name, column in flags.items():
        if column in out.columns:
            # Errored rows carry no flags; treating their NaN as False keeps the
            # rate a share of all recordings attempted, which is the honest
            # denominator for a reliability figure.
            out[column] = out[column].fillna(False).astype(bool)
            aggregations[name] = (column, "mean")

    return (
        out.groupby(list(by), observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(list(by))
    )


def direction_response(frame, metric="lf_level_db_gc", by=("site",), sectors=8,
                       min_count=5, min_knots=10.0):
    """
    Break the wind response down by the direction the wind came from.

    A site sheltered by a ridge to the west is loud in an easterly and quiet in
    a westerly. Averaging over direction hides that, and it is exactly the
    information that would inform where to put the recorder next season.

    Only recordings above `min_knots` are used: in a calm, the recorded
    direction is close to meaningless and would only add noise.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        metric (str): The acoustic column to summarize.
        by (sequence of str): Grouping columns.
        sectors (int): Compass sectors to split into.
        min_count (int): Sectors with fewer recordings are dropped.
        min_knots (float): Ignore recordings below this wind speed.

    Returns:
        pandas.DataFrame: ``*by``, ``sector_deg`` (the sector's centre),
        ``n``, ``mean`` and ``median``.

    Raises:
        ValueError: If the wind direction column is missing.
    """
    if "wind_dir_deg" not in frame.columns:
        raise ValueError("frame has no 'wind_dir_deg' column")

    width = 360.0 / sectors
    windy = frame[frame["wind_knots"].fillna(0) >= min_knots].dropna(
        subset=[metric, "wind_dir_deg"]
    ).copy()
    # Offset by half a sector so that north lands mid-sector rather than on a
    # boundary, which would split northerlies across two bins.
    windy["sector_deg"] = (
        ((windy["wind_dir_deg"] + width / 2) % 360) // width * width
    )

    table = (
        windy.groupby(list(by) + ["sector_deg"], observed=True)
        .agg(n=(metric, "size"), mean=(metric, "mean"), median=(metric, "median"))
        .reset_index()
    )
    return table[table["n"] >= min_count].reset_index(drop=True)


#: Quantile edges for rank-based wind bands, and their labels.
WIND_PERCENTILE_EDGES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
WIND_PERCENTILE_LABELS = ("calmest 25%", "25-50%", "50-75%", "75-90%", "windiest 10%")


def add_wind_percentile_bands(frame, by=("season",), column="wind_knots",
                              edges=WIND_PERCENTILE_EDGES,
                              labels=WIND_PERCENTILE_LABELS,
                              target="wind_pct_band"):
    """
    Band recordings by how windy they were *relative to their own season*.

    A fallback for the scale break described in `meteo.WIND_SCALE_BREAK`: the
    station's knots are not one quantity across the whole record, so absolute
    bands cannot span it. Ranking within each season sidesteps the units
    entirely.

    The assumption this buys is not free. Rank-matching assumes the *true*
    distribution of wind was similar between the seasons being compared. If one
    season really was calmer, that difference is erased along with the
    instrument change, and a site that merely got luckier with the weather will
    look better deployed. Use this as a sensitivity check against the
    absolute-band result, never as the primary answer.

    Args:
        frame (pandas.DataFrame): Recordings joined to weather.
        by (sequence of str): Columns defining the group to rank within.
        column (str): The wind column to rank.
        edges (sequence of float): Quantile cut points, 0 to 1.
        labels (sequence of str): One fewer than `edges`.
        target (str): Name of the column to add.

    Returns:
        pandas.DataFrame: A copy with `target` added, as an ordered categorical.
        Rows with no wind reading get NaN.

    Raises:
        ValueError: If `column` is missing or the labels do not match the edges.
    """
    if column not in frame.columns:
        raise ValueError(f"frame has no column {column!r}")
    if len(labels) != len(edges) - 1:
        raise ValueError(
            f"got {len(edges)} edges and {len(labels)} labels; "
            "labels must be one shorter"
        )

    out = frame.copy()
    out[target] = pd.Series(pd.NA, index=out.index, dtype="object")

    for _, group in out.groupby(list(by), observed=True):
        values = group[column]
        if values.notna().sum() < len(labels):
            # Too few readings to cut into this many bands; leave them null
            # rather than inventing boundaries from a handful of points.
            continue
        banded = pd.qcut(
            values, q=list(edges), labels=list(labels), duplicates="drop"
        )
        out.loc[group.index, target] = banded.astype(object)

    out[target] = pd.Categorical(out[target], categories=list(labels), ordered=True)
    return out
