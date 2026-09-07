"""
Weather observations from the Frei station, aligned to recording timestamps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The station logs local time; recordings are stamped UTC. Local + 3 h = UTC.
STATION_UTC_OFFSET_HOURS = -3

#: The wind column is not on one scale for the whole record. Across the 184-day
#: winter gap before the 2023-24 season it steps up: mean 11.8 -> 18.3 kt,
#: 95th percentile 20 -> 33 kt. What settles it is the ceiling — in 32,944
#: readings before the break not one exceeds 34 knots, while 2,146 of the 51,968
#: after it do. Fourteen consecutive months without a single reading above
#: 34 kt is not weather in the South Shetlands.
#:
#: Temperature, humidity and pressure are unchanged across the same boundary, so
#: this is the wind instrument or its processing, not a station move. The ratio
#: of means (1.55) and of 95th percentiles (1.65) both sit in the range expected
#: between a 10-minute mean and a 10-minute maximum, which is the most likely
#: explanation: the workbook documents `IntVM` as a 10-minute maximum, and the
#: later data behaves like one.
#:
#: The consequence is unavoidable: **knots either side of this date are
#: different quantities**. Seasons 2023-24 and 2024-25 can be compared directly;
#: 2022-23 cannot be put on the same axis as either. `report.add_wind_percentile_bands`
#: offers a rank-based fallback for spanning it, with its own assumption.
WIND_SCALE_BREAK = pd.Timestamp("2023-10-01", tz="UTC")

#: Spanish month names as they appear in the workbook.
MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

#: Workbook column names mapped to the names used downstream.
COLUMN_MAP = {
    "Taire": "air_temp_c",
    "DirVM": "wind_dir_deg",
    "IntVM": "wind_knots",
    "HR": "humidity_pct",
    "PNMM": "pressure_hpa",
}

#: Beaufort-style bands, in knots, for grouping recordings by wind strength.
#: Splitting at 10-knot steps keeps every band populated in this dataset; the
#: station's median is 15 knots and its maximum 65.
WIND_BANDS = [0, 5, 10, 15, 20, 25, 30, 40, 100]
WIND_BAND_LABELS = [
    "0-5 kt", "5-10 kt", "10-15 kt", "15-20 kt",
    "20-25 kt", "25-30 kt", "30-40 kt", "40+ kt",
]


def _clock_parts(value):
    """
    Read (hour, minute) from a time-of-day cell.

    Excel hands the same column back as `datetime.time` from one writer and as
    a ``"16:50:00"`` string from another, so both are accepted. Anything
    unreadable becomes NaN and the row is dropped downstream.

    Args:
        value: A `datetime.time`, a datetime, or a clock string.

    Returns:
        tuple of (float, float): Hour and minute, or ``(nan, nan)``.
    """
    hour = getattr(value, "hour", None)
    if hour is not None:
        return float(hour), float(getattr(value, "minute", 0))

    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return np.nan, np.nan
    return float(parsed.hour), float(parsed.minute)


def load_meteo(path, sheet="Datos"):
    """
    Read the station workbook into a UTC-indexed frame.

    Args:
        path (str | Path): The ``Meteo_*.xlsx`` workbook.
        sheet (str): Sheet holding the observations.

    Returns:
        pandas.DataFrame: Columns ``timestamp_utc`` (tz-aware), ``air_temp_c``,
        ``wind_dir_deg``, ``wind_knots``, ``wind_ms``, ``humidity_pct``,
        ``pressure_hpa`` and ``wind_band``, sorted by time with unparseable rows
        dropped.

    Raises:
        ValueError: If the sheet is missing the date or time columns.
    """
    frame = pd.read_excel(Path(path), sheet_name=sheet)

    required = {"Dia", "Mes", "Año", "Hora"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"meteo sheet is missing columns: {sorted(missing)}")

    month = (
        frame["Mes"].astype(str).str.strip().str.lower().map(MONTHS_ES)
    )
    clock = frame["Hora"].map(_clock_parts)
    hour = clock.map(lambda parts: parts[0])
    minute = clock.map(lambda parts: parts[1])

    local = pd.to_datetime(
        pd.DataFrame(
            {
                "year": frame["Año"],
                "month": month,
                "day": frame["Dia"],
                "hour": hour,
                "minute": minute,
            }
        ),
        errors="coerce",
    )

    out = pd.DataFrame({"timestamp_local": local})
    for source, target in COLUMN_MAP.items():
        out[target] = pd.to_numeric(frame.get(source), errors="coerce")

    out = out.dropna(subset=["timestamp_local"]).copy()
    out["timestamp_utc"] = (
        out["timestamp_local"].dt.tz_localize("UTC")
        - pd.Timedelta(hours=STATION_UTC_OFFSET_HOURS)
    )
    out["wind_ms"] = out["wind_knots"] * 0.514444
    out["wind_band"] = pd.cut(
        out["wind_knots"], bins=WIND_BANDS, labels=WIND_BAND_LABELS, right=False
    )

    return (
        out.drop(columns=["timestamp_local"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )


def join_meteo(recordings, meteo, tolerance="10min", time_column="timestamp_utc"):
    """
    Attach the nearest weather observation to each recording.

    A nearest-match join rather than a resample: recordings do not always land
    on the station's 10-minute grid, and a recording with no observation within
    `tolerance` should carry nulls rather than a stale reading from hours away.

    Args:
        recordings (pandas.DataFrame): Must carry a tz-aware `time_column`.
        meteo (pandas.DataFrame): Output of `load_meteo`.
        tolerance (str): Maximum gap to accept, as a pandas offset string.
        time_column (str): The recordings' timestamp column.

    Returns:
        pandas.DataFrame: `recordings` with the weather columns added, plus
        ``meteo_gap_s`` giving the distance in seconds to the matched
        observation. Rows are returned in time order.

    Raises:
        ValueError: If `time_column` is missing or not timezone-aware.
    """
    if time_column not in recordings.columns:
        raise ValueError(f"recordings has no column {time_column!r}")

    left = recordings.dropna(subset=[time_column]).sort_values(time_column).copy()
    if left.empty:
        raise ValueError(f"no rows with a usable {time_column}")
    if left[time_column].dt.tz is None:
        raise ValueError(f"{time_column} must be timezone-aware")

    right = meteo.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").copy()
    right["_meteo_time"] = right["timestamp_utc"]

    merged = pd.merge_asof(
        left,
        right.drop(columns=["timestamp_utc"]),
        left_on=time_column,
        right_on="_meteo_time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    merged["meteo_gap_s"] = (
        merged[time_column] - merged["_meteo_time"]
    ).dt.total_seconds().abs()
    return merged.drop(columns=["_meteo_time"])


def verify_time_alignment(recordings, meteo, metric="lf_level_db",
                          offsets_h=range(-6, 7), time_column="timestamp_utc"):
    """
    Re-derive the clock offset between recordings and the station from the data.

    Shifts the recordings against the weather by a range of whole hours and
    correlates `metric` with wind speed at each shift. If the assumed offset is
    right, the correlation peaks at 0. Any other peak means the assumption is
    wrong, and the peak says by how much.

    Args:
        recordings (pandas.DataFrame): Recordings carrying `metric` and a
            tz-aware `time_column`. May already be joined to weather; any
            existing weather columns are dropped before re-joining.
        meteo (pandas.DataFrame): Output of `load_meteo`.
        metric (str): The acoustic column to correlate against wind speed.
        offsets_h (iterable of int): Candidate shifts in hours.
        time_column (str): The recordings' timestamp column.

    Returns:
        pandas.DataFrame: ``offset_h``, ``correlation`` and ``n`` per candidate,
        ordered by offset. The best shift is the row with the highest
        correlation.
    """
    # The natural thing to hand this function is a frame that has already been
    # joined to weather. Re-joining one would collide on every weather column
    # and pandas would silently suffix them away, so they are dropped first.
    weather_columns = [c for c in meteo.columns if c != "timestamp_utc"]
    base = recordings.drop(
        columns=[c for c in weather_columns + ["meteo_gap_s"] if c in recordings],
    )

    rows = []
    for offset in offsets_h:
        shifted = base.copy()
        shifted[time_column] = shifted[time_column] + pd.Timedelta(hours=offset)
        joined = join_meteo(shifted, meteo, time_column=time_column)
        pair = joined[[metric, "wind_knots"]].dropna()
        rows.append(
            {
                "offset_h": offset,
                "correlation": (
                    pair[metric].corr(pair["wind_knots"]) if len(pair) > 2 else np.nan
                ),
                "n": len(pair),
            }
        )
    return pd.DataFrame(rows)
