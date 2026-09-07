"""Tests for the weather-controlled aggregations."""

import numpy as np
import pandas as pd
import pytest

from soundsanity.report import (
    add_wind_percentile_bands,
    apply_gain_correction,
    deployment_quality,
    direction_response,
    standardize_by_wind,
    wind_response,
    wind_weights,
)

BANDS = ["0-5 kt", "5-10 kt", "10-15 kt", "15-20 kt"]


def _frame(rows):
    """Build an analyzed-and-joined frame from (season, site, gain, band, level) tuples."""
    frame = pd.DataFrame(
        rows, columns=["season", "site", "gain", "wind_band", "lf_level_db"]
    )
    frame["wind_band"] = pd.Categorical(frame["wind_band"], categories=BANDS)
    frame["status"] = "NOISY"
    frame["is_windy"] = frame["lf_level_db"] > -40
    return frame


class TestApplyGainCorrection:
    def test_normalizes_levels_to_the_medium_setting(self):
        frame = _frame([("2022-23", "Frei", "medium-high", "5-10 kt", -30.0)])
        out = apply_gain_correction(frame)
        # medium-high sits 6 dB above medium, so the corrected level drops 6 dB.
        assert out["lf_level_db_gc"].iloc[0] == pytest.approx(-36.0)

    def test_medium_is_unchanged(self):
        frame = _frame([("2022-23", "Frei", "medium", "5-10 kt", -30.0)])
        assert apply_gain_correction(frame)["lf_level_db_gc"].iloc[0] == -30.0

    def test_unknown_gain_becomes_null_rather_than_uncorrected(self):
        frame = _frame([("2022-23", "Frei", None, "5-10 kt", -30.0)])
        out = apply_gain_correction(frame)
        assert pd.isna(out["lf_level_db_gc"].iloc[0])

    def test_leaves_the_original_column_intact(self):
        frame = _frame([("2022-23", "Frei", "high", "5-10 kt", -30.0)])
        assert apply_gain_correction(frame)["lf_level_db"].iloc[0] == -30.0

    def test_skips_columns_that_are_absent(self):
        frame = _frame([("2022-23", "Frei", "medium", "5-10 kt", -30.0)])
        out = apply_gain_correction(frame, columns=("lf_level_db", "not_here_db"))
        assert "not_here_db_gc" not in out.columns

    def test_requires_a_gain_column(self):
        with pytest.raises(ValueError, match="no 'gain' column"):
            apply_gain_correction(pd.DataFrame({"lf_level_db": [-30.0]}))


class TestWindResponse:
    def test_summarizes_per_group_and_band(self):
        frame = _frame(
            [("2022-23", "Frei", "medium", "5-10 kt", -40.0)] * 6
            + [("2022-23", "Frei", "medium", "15-20 kt", -20.0)] * 6
        )
        table = wind_response(frame, metric="lf_level_db", by=("site",))
        assert len(table) == 2
        assert table.set_index("wind_band")["mean"]["15-20 kt"] == -20.0

    def test_drops_thinly_populated_bands(self):
        frame = _frame(
            [("2022-23", "Frei", "medium", "5-10 kt", -40.0)] * 6
            + [("2022-23", "Frei", "medium", "15-20 kt", -20.0)] * 2
        )
        table = wind_response(frame, metric="lf_level_db", by=("site",), min_count=5)
        assert table["wind_band"].tolist() == ["5-10 kt"]

    def test_reports_the_windy_share(self):
        frame = _frame(
            [("2022-23", "Frei", "medium", "5-10 kt", -20.0)] * 5
            + [("2022-23", "Frei", "medium", "5-10 kt", -60.0)] * 5
        )
        table = wind_response(frame, metric="lf_level_db", by=("site",))
        assert table["windy_rate"].iloc[0] == pytest.approx(0.5)

    def test_rejects_a_missing_metric(self):
        with pytest.raises(ValueError, match="no column"):
            wind_response(_frame([]), metric="nope")


class TestWindWeights:
    def test_weights_sum_to_one(self):
        frame = _frame(
            [("2022-23", "Frei", "medium", "5-10 kt", -40.0)] * 3
            + [("2022-23", "Frei", "medium", "15-20 kt", -20.0)] * 1
        )
        weights = wind_weights(frame)
        assert weights.sum() == pytest.approx(1.0)
        assert weights["5-10 kt"] == pytest.approx(0.75)


class TestStandardizeByWind:
    def test_two_seasons_with_identical_response_score_equally(self):
        # Season B saw far more wind than A but behaved identically at each
        # strength. Standardizing must erase the difference.
        rows = []
        for _ in range(20):
            rows.append(("A", "Frei", "medium", "5-10 kt", -50.0))
        for _ in range(5):
            rows.append(("A", "Frei", "medium", "15-20 kt", -30.0))
        for _ in range(5):
            rows.append(("B", "Frei", "medium", "5-10 kt", -50.0))
        for _ in range(20):
            rows.append(("B", "Frei", "medium", "15-20 kt", -30.0))

        table = standardize_by_wind(_frame(rows), metric="lf_level_db", by=("season",))
        standardized = table.set_index("season")["standardized"]
        assert standardized["A"] == pytest.approx(standardized["B"])

        # The raw means, by contrast, differ a great deal — which is the trap.
        raw = table.set_index("season")["raw_mean"]
        assert abs(raw["A"] - raw["B"]) > 10

    def test_a_genuinely_quieter_season_still_scores_lower(self):
        rows = (
            [("A", "Frei", "medium", "5-10 kt", -50.0)] * 10
            + [("A", "Frei", "medium", "15-20 kt", -30.0)] * 10
            + [("B", "Frei", "medium", "5-10 kt", -56.0)] * 10
            + [("B", "Frei", "medium", "15-20 kt", -36.0)] * 10
        )
        table = standardize_by_wind(_frame(rows), metric="lf_level_db", by=("season",))
        standardized = table.set_index("season")["standardized"]
        assert standardized["B"] == pytest.approx(standardized["A"] - 6.0)

    def test_coverage_flags_a_group_seen_in_few_conditions(self):
        rows = (
            [("A", "Frei", "medium", "5-10 kt", -50.0)] * 10
            + [("A", "Frei", "medium", "15-20 kt", -30.0)] * 10
            + [("B", "Frei", "medium", "5-10 kt", -50.0)] * 10
        )
        table = standardize_by_wind(_frame(rows), metric="lf_level_db", by=("season",))
        coverage = table.set_index("season")["coverage"]
        assert coverage["A"] == pytest.approx(1.0)
        assert coverage["B"] < 1.0

    def test_accepts_externally_supplied_weights(self):
        rows = (
            [("A", "Frei", "medium", "5-10 kt", -50.0)] * 10
            + [("A", "Frei", "medium", "15-20 kt", -30.0)] * 10
        )
        weights = pd.Series({"5-10 kt": 0.0, "15-20 kt": 1.0})
        table = standardize_by_wind(
            _frame(rows), metric="lf_level_db", by=("season",), weights=weights
        )
        assert table["standardized"].iloc[0] == pytest.approx(-30.0)


class TestDeploymentQuality:
    def test_reports_rates_per_group(self):
        frame = _frame([("A", "Frei", "medium", "5-10 kt", -20.0)] * 4)
        frame["is_saturated"] = [True, False, False, False]
        table = deployment_quality(frame, by=("season", "site"))
        assert table["saturated_rate"].iloc[0] == pytest.approx(0.25)
        assert table["n"].iloc[0] == 4

    def test_errored_rows_count_in_the_denominator(self):
        frame = _frame([("A", "Frei", "medium", "5-10 kt", -20.0)] * 4)
        frame.loc[0, "status"] = "ERROR"
        frame["is_saturated"] = [np.nan, False, False, True]
        table = deployment_quality(frame, by=("season",))
        assert table["error_rate"].iloc[0] == pytest.approx(0.25)
        # The errored row is not saturated, but it is still a file that was
        # attempted, so the rate is 1/4 rather than 1/3.
        assert table["saturated_rate"].iloc[0] == pytest.approx(0.25)


class TestDirectionResponse:
    def test_bins_by_compass_sector(self):
        frame = _frame([("A", "Frei", "medium", "15-20 kt", -30.0)] * 10)
        frame["wind_knots"] = 18.0
        frame["wind_dir_deg"] = [0, 2, 358, 1, 359, 180, 181, 179, 182, 178]
        table = direction_response(frame, metric="lf_level_db", by=("site",))
        # Northerlies straddling 360 must land in one sector, not two.
        assert sorted(table["sector_deg"]) == [0.0, 180.0]
        assert table["n"].tolist() == [5, 5]

    def test_calm_recordings_are_excluded(self):
        frame = _frame([("A", "Frei", "medium", "0-5 kt", -60.0)] * 10)
        frame["wind_knots"] = 2.0
        frame["wind_dir_deg"] = 90.0
        assert direction_response(frame, metric="lf_level_db", by=("site",)).empty

    def test_requires_a_direction_column(self):
        with pytest.raises(ValueError, match="no 'wind_dir_deg'"):
            direction_response(_frame([]), metric="lf_level_db")


class TestAddWindPercentileBands:
    """Rank-based bands, the fallback for the station's mid-record scale change."""

    def _two_seasons(self):
        # Season B's readings are on a different scale entirely, as the station's
        # are either side of the 2023 winter.
        return pd.DataFrame(
            {
                "season": ["A"] * 20 + ["B"] * 20,
                "wind_knots": list(range(1, 21)) + list(range(31, 51)),
            }
        )

    def test_ranks_within_each_season(self):
        out = add_wind_percentile_bands(self._two_seasons())
        calm_a = out[(out["season"] == "A") & (out["wind_pct_band"] == "calmest 25%")]
        calm_b = out[(out["season"] == "B") & (out["wind_pct_band"] == "calmest 25%")]
        # Both seasons contribute their own calmest quarter despite sharing no
        # values at all.
        assert len(calm_a) == len(calm_b) == 5
        assert calm_a["wind_knots"].max() < calm_b["wind_knots"].min()

    def test_band_is_ordered(self):
        out = add_wind_percentile_bands(self._two_seasons())
        assert out["wind_pct_band"].cat.ordered
        assert list(out["wind_pct_band"].cat.categories)[0] == "calmest 25%"

    def test_missing_wind_readings_stay_null(self):
        frame = self._two_seasons()
        frame.loc[0, "wind_knots"] = np.nan
        out = add_wind_percentile_bands(frame)
        assert pd.isna(out.loc[0, "wind_pct_band"])

    def test_a_group_with_too_few_readings_is_left_null(self):
        frame = pd.DataFrame({"season": ["A", "A"], "wind_knots": [5.0, 9.0]})
        out = add_wind_percentile_bands(frame)
        assert out["wind_pct_band"].isna().all()

    def test_standardizing_on_percentile_bands_works(self):
        frame = self._two_seasons()
        frame["lf_level_db"] = np.where(frame["season"] == "A", -50.0, -56.0)
        frame["gain"] = "medium"
        frame["status"] = "NOISY"
        banded = add_wind_percentile_bands(frame)
        table = standardize_by_wind(
            banded, metric="lf_level_db", by=("season",),
            band_column="wind_pct_band", min_count=2,
        )
        # Same rank distribution, 6 dB quieter: the gap must survive.
        gap = table.set_index("season")["standardized"]
        assert gap["B"] == pytest.approx(gap["A"] - 6.0)

    def test_rejects_mismatched_labels(self):
        with pytest.raises(ValueError, match="one shorter"):
            add_wind_percentile_bands(
                self._two_seasons(), edges=(0.0, 0.5, 1.0), labels=("only one",)
            )

    def test_rejects_a_missing_column(self):
        with pytest.raises(ValueError, match="no column"):
            add_wind_percentile_bands(self._two_seasons(), column="gusts")
