"""Tests for loading station weather and aligning it to recording timestamps."""

from datetime import time as clock_time

import pandas as pd
import pytest

from soundsanity.meteo import (
    STATION_UTC_OFFSET_HOURS,
    join_meteo,
    load_meteo,
    verify_time_alignment,
)


@pytest.fixture
def meteo_workbook(tmp_path):
    """Write a workbook shaped like the station's export."""

    def _write(rows=None, name="meteo.xlsx"):
        rows = rows if rows is not None else [
            {"Dia": 10, "Mes": "Diciembre", "Año": 2023, "Hora": clock_time(16, 50),
             "Taire": 1.5, "DirVM": 270.0, "IntVM": 22.0, "HR": 88.0, "PNMM": 990.0},
            {"Dia": 10, "Mes": "Diciembre", "Año": 2023, "Hora": clock_time(17, 0),
             "Taire": 1.6, "DirVM": 268.0, "IntVM": 25.0, "HR": 87.0, "PNMM": 989.0},
            {"Dia": 10, "Mes": "diciembre", "Año": 2023, "Hora": clock_time(17, 10),
             "Taire": 1.7, "DirVM": 265.0, "IntVM": 3.0, "HR": 86.0, "PNMM": 989.0},
        ]
        path = tmp_path / name
        pd.DataFrame(rows).to_excel(path, sheet_name="Datos", index=False)
        return path

    return _write


@pytest.fixture
def recordings():
    """Two recordings stamped in UTC, as the inventory produces them."""
    return pd.DataFrame(
        {
            "path": ["a.wav", "b.wav"],
            "timestamp_utc": pd.to_datetime(
                ["2023-12-10 19:50:00", "2023-12-10 20:10:00"], utc=True
            ),
            "lf_level_db": [-25.0, -55.0],
        }
    )


class TestLoadMeteo:
    def test_shifts_local_station_time_to_utc(self, meteo_workbook):
        meteo = load_meteo(meteo_workbook())
        # 16:50 local at UTC-3 is 19:50 UTC.
        assert meteo["timestamp_utc"].iloc[0] == pd.Timestamp(
            "2023-12-10 19:50:00", tz="UTC"
        )
        assert STATION_UTC_OFFSET_HOURS == -3

    def test_month_names_are_case_insensitive(self, meteo_workbook):
        assert len(load_meteo(meteo_workbook())) == 3

    def test_renames_columns_and_derives_wind_units(self, meteo_workbook):
        meteo = load_meteo(meteo_workbook())
        assert {"air_temp_c", "wind_knots", "wind_ms", "wind_dir_deg",
                "humidity_pct", "pressure_hpa"} <= set(meteo.columns)
        assert meteo["wind_ms"].iloc[0] == pytest.approx(22.0 * 0.514444)

    def test_assigns_wind_bands(self, meteo_workbook):
        meteo = load_meteo(meteo_workbook())
        assert str(meteo["wind_band"].iloc[0]) == "20-25 kt"
        assert str(meteo["wind_band"].iloc[2]) == "0-5 kt"

    def test_drops_rows_with_an_unparseable_date(self, meteo_workbook):
        rows = [
            {"Dia": 10, "Mes": "Diciembre", "Año": 2023, "Hora": clock_time(16, 50),
             "Taire": 1.5, "DirVM": 270.0, "IntVM": 22.0, "HR": 88.0, "PNMM": 990.0},
            {"Dia": 99, "Mes": "Brumaire", "Año": 2023, "Hora": clock_time(17, 0),
             "Taire": 1.6, "DirVM": 268.0, "IntVM": 25.0, "HR": 87.0, "PNMM": 989.0},
        ]
        assert len(load_meteo(meteo_workbook(rows=rows))) == 1

    def test_missing_readings_stay_null_rather_than_zero(self, meteo_workbook):
        rows = [
            {"Dia": 10, "Mes": "Diciembre", "Año": 2023, "Hora": clock_time(16, 50),
             "Taire": 1.5, "DirVM": None, "IntVM": None, "HR": 88.0, "PNMM": 990.0},
        ]
        meteo = load_meteo(meteo_workbook(rows=rows))
        assert pd.isna(meteo["wind_knots"].iloc[0])

    def test_rejects_a_sheet_without_date_columns(self, tmp_path):
        path = tmp_path / "wrong.xlsx"
        pd.DataFrame({"foo": [1]}).to_excel(path, sheet_name="Datos", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_meteo(path)


class TestJoinMeteo:
    def test_attaches_the_nearest_observation(self, meteo_workbook, recordings):
        joined = join_meteo(recordings, load_meteo(meteo_workbook()))
        assert joined["wind_knots"].tolist() == [22.0, 3.0]

    def test_reports_the_gap_to_the_matched_reading(self, meteo_workbook, recordings):
        joined = join_meteo(recordings, load_meteo(meteo_workbook()))
        assert joined["meteo_gap_s"].iloc[0] == 0.0

    def test_leaves_unmatched_recordings_null(self, meteo_workbook):
        far = pd.DataFrame(
            {
                "path": ["z.wav"],
                "timestamp_utc": pd.to_datetime(["2024-06-01 00:00:00"], utc=True),
            }
        )
        joined = join_meteo(far, load_meteo(meteo_workbook()))
        # A reading six months away is worse than no reading at all.
        assert pd.isna(joined["wind_knots"].iloc[0])

    def test_tolerance_is_respected(self, meteo_workbook, recordings):
        # One recording sits exactly on the station's 10-minute grid, the other
        # five minutes off it; a one-second tolerance should keep only the first.
        off_grid = recordings.copy()
        off_grid.loc[1, "timestamp_utc"] += pd.Timedelta(minutes=5)
        joined = join_meteo(off_grid, load_meteo(meteo_workbook()), tolerance="1s")
        assert joined["wind_knots"].notna().sum() == 1

    def test_rejects_naive_timestamps(self, meteo_workbook, recordings):
        naive = recordings.assign(
            timestamp_utc=recordings["timestamp_utc"].dt.tz_localize(None)
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            join_meteo(naive, load_meteo(meteo_workbook()))

    def test_rejects_a_missing_time_column(self, meteo_workbook, recordings):
        with pytest.raises(ValueError, match="no column"):
            join_meteo(recordings, load_meteo(meteo_workbook()), time_column="when")


class TestVerifyTimeAlignment:
    def test_peaks_at_zero_when_the_offset_is_right(self, meteo_workbook):
        # Loud recordings placed on the windy readings, quiet on the calm one.
        meteo = load_meteo(meteo_workbook())
        aligned = pd.DataFrame(
            {
                "timestamp_utc": meteo["timestamp_utc"],
                "lf_level_db": [-25.0, -20.0, -60.0],
            }
        )
        table = verify_time_alignment(
            aligned, meteo, offsets_h=range(-2, 3)
        )
        assert table.loc[table["correlation"].idxmax(), "offset_h"] == 0

    def test_reports_one_row_per_candidate_offset(self, meteo_workbook, recordings):
        table = verify_time_alignment(
            recordings, load_meteo(meteo_workbook()), offsets_h=range(-3, 4)
        )
        assert list(table["offset_h"]) == list(range(-3, 4))
        assert {"correlation", "n"} <= set(table.columns)


class TestVerifyTimeAlignmentOnJoinedInput:
    def test_accepts_a_frame_that_is_already_joined(self, meteo_workbook):
        # The obvious way to call this is with the frame you already built, which
        # carries wind_knots. Re-joining that without dropping it first collides
        # on every weather column.
        meteo = load_meteo(meteo_workbook())
        recordings = pd.DataFrame(
            {
                "timestamp_utc": meteo["timestamp_utc"],
                "lf_level_db": [-25.0, -20.0, -60.0],
            }
        )
        already_joined = join_meteo(recordings, meteo)
        assert "wind_knots" in already_joined.columns

        table = verify_time_alignment(already_joined, meteo, offsets_h=range(-1, 2))
        # Only the zero shift lands on this fixture's half-hour of weather; the
        # point is that it correlates at all rather than colliding into NaN.
        at_zero = table.set_index("offset_h").loc[0]
        assert at_zero["correlation"] == pytest.approx(1.0, abs=0.01)
        assert at_zero["n"] == 3
