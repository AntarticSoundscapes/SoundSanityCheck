"""Tests for the diagnostic checks in soundsanity.analysis."""

import copy

import numpy as np
import pytest

import soundsanity as ss
from soundsanity.analysis import merge_clicks, merge_intervals


class TestMergeIntervals:
    def test_empty_input(self):
        assert merge_intervals([]) == []

    def test_single_interval_is_preserved(self):
        assert merge_intervals([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_overlapping_intervals_are_merged(self):
        assert merge_intervals([(0.0, 1.0), (0.5, 2.0)]) == [(0.0, 2.0)]

    def test_disjoint_intervals_are_kept_apart(self):
        assert merge_intervals([(0.0, 1.0), (3.0, 4.0)]) == [(0.0, 1.0), (3.0, 4.0)]

    def test_unsorted_input_is_sorted_first(self):
        assert merge_intervals([(3.0, 4.0), (0.0, 1.0)]) == [(0.0, 1.0), (3.0, 4.0)]

    def test_gap_tolerance_bridges_adjacent_intervals(self):
        intervals = [(0.0, 1.0), (1.05, 2.0)]
        assert merge_intervals(intervals, gap_tolerance=0.0) == intervals
        assert merge_intervals(intervals, gap_tolerance=0.1) == [(0.0, 2.0)]

    def test_nested_interval_does_not_shrink_the_parent(self):
        assert merge_intervals([(0.0, 5.0), (1.0, 2.0)]) == [(0.0, 5.0)]


class TestMergeClicks:
    def test_empty_input(self):
        assert merge_clicks([]) == []

    def test_close_clicks_collapse_to_their_mean(self):
        merged = merge_clicks([1.00, 1.005, 1.01], tolerance=0.01)
        assert len(merged) == 1
        assert merged[0] == pytest.approx(1.005)

    def test_distant_clicks_stay_separate(self):
        assert merge_clicks([1.0, 5.0], tolerance=0.01) == [1.0, 5.0]

    def test_unsorted_input_is_sorted_first(self):
        assert merge_clicks([5.0, 1.0], tolerance=0.01) == [1.0, 5.0]

    def test_tolerance_is_measured_against_the_previous_click(self):
        # Each step is within tolerance of the last, so the whole chain groups.
        merged = merge_clicks([1.0, 1.008, 1.016, 1.024], tolerance=0.01)
        assert len(merged) == 1


class TestDetectClicks:
    def test_clean_audio_is_not_clicky(self, clean_audio, sample_rate):
        result = ss.detect_clicks(clean_audio, sample_rate)
        assert result["is_clicky"] is False
        assert result["clicks_rate"] <= ss.DEFAULT_CONFIG["max_clicks_rate"]

    def test_injected_clicks_are_detected(self, clean_audio, sample_rate):
        degraded, _ = ss.add_clicks(clean_audio, num_clicks=100)
        result = ss.detect_clicks(degraded, sample_rate)
        assert result["is_clicky"] is True
        assert result["clicks_count"] > 0
        assert len(result["click_timestamps"]) == result["clicks_count"]

    def test_empty_audio_returns_zeroed_metrics(self, empty_audio, sample_rate):
        assert ss.detect_clicks(empty_audio, sample_rate) == {
            "clicks_count": 0,
            "clicks_rate": 0.0,
            "click_timestamps": [],
            "is_clicky": False,
        }

    def test_rate_threshold_is_configurable(self, clean_audio, sample_rate):
        degraded, _ = ss.add_clicks(clean_audio, num_clicks=100)
        relaxed = ss.detect_clicks(degraded, sample_rate, {"max_clicks_rate": 1e6})
        assert relaxed["is_clicky"] is False
        assert relaxed["clicks_count"] > 0


class TestDetectSaturation:
    def test_clean_audio_is_not_saturated(self, clean_audio, sample_rate):
        result = ss.detect_saturation(clean_audio, sample_rate)
        assert result["is_saturated"] is False
        assert result["saturation_ratio"] == pytest.approx(0.0)

    def test_clipped_audio_is_flagged(self, clean_audio, sample_rate):
        clipped = ss.add_clipping(clean_audio, clip_threshold=0.01)
        result = ss.detect_saturation(clipped, sample_rate)
        assert result["is_saturated"] is True
        assert result["saturation_duration"] > 0.0
        assert 0.0 < result["saturation_ratio"] <= 1.0

    def test_intervals_are_ordered_and_non_overlapping(self, clean_audio, sample_rate):
        clipped = ss.add_clipping(clean_audio, clip_threshold=0.01)
        intervals = ss.detect_saturation(clipped, sample_rate)["saturation_intervals"]
        for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
            assert next_start > prev_end

    def test_empty_audio_returns_zeroed_metrics(self, empty_audio, sample_rate):
        assert ss.detect_saturation(empty_audio, sample_rate) == {
            "saturation_intervals": [],
            "saturation_duration": 0.0,
            "saturation_ratio": 0.0,
            "is_saturated": False,
        }


class TestDetectSilence:
    def test_clean_audio_is_not_silent(self, clean_audio, sample_rate):
        result = ss.detect_silence(clean_audio, sample_rate)
        assert result["is_silent"] is False
        assert 0.0 <= result["silence_ratio"] <= 1.0

    def test_attenuated_audio_is_flagged_as_silent(self, clean_audio, sample_rate):
        silent = ss.make_silent(clean_audio, level_db=-90)
        result = ss.detect_silence(silent, sample_rate)
        assert result["is_silent"] is True
        assert result["silence_ratio"] == pytest.approx(1.0)

    def test_empty_audio_counts_as_silent(self, empty_audio, sample_rate):
        assert ss.detect_silence(empty_audio, sample_rate) == {
            "silence_ratio": 1.0,
            "is_silent": True,
        }

    def test_threshold_is_configurable(self, clean_audio, sample_rate):
        # A -200 dB floor is below anything present, so nothing reads as silent.
        result = ss.detect_silence(clean_audio, sample_rate, {"silence_threshold_db": -200.0})
        assert result["silence_ratio"] == pytest.approx(0.0)
        assert result["is_silent"] is False


class TestEstimateNoise:
    def test_clean_audio_has_headroom(self, clean_audio, sample_rate):
        result = ss.estimate_noise(clean_audio, sample_rate)
        assert result["is_noisy"] is False
        assert result["noise_floor_db"] < ss.DEFAULT_CONFIG["max_noise_floor_db"]
        assert result["snr_db"] > ss.DEFAULT_CONFIG["min_snr_db"]

    def test_snr_is_the_gap_between_active_level_and_noise_floor(self, clean_audio, sample_rate):
        result = ss.estimate_noise(clean_audio, sample_rate)
        expected = result["active_level_db"] - result["noise_floor_db"]
        assert result["snr_db"] == pytest.approx(expected)

    def test_added_noise_collapses_the_snr(self, clean_audio, sample_rate):
        baseline = ss.estimate_noise(clean_audio, sample_rate)
        noisy = ss.estimate_noise(ss.add_noise(clean_audio, target_snr_db=0), sample_rate)
        assert noisy["is_noisy"] is True
        assert noisy["snr_db"] < baseline["snr_db"]
        assert noisy["noise_floor_db"] > baseline["noise_floor_db"]

    def test_empty_audio_returns_floor_values(self, empty_audio, sample_rate):
        assert ss.estimate_noise(empty_audio, sample_rate) == {
            "rms_db": -100.0,
            "noise_floor_db": -100.0,
            "active_level_db": -100.0,
            "snr_db": 0.0,
            "is_noisy": True,
        }

    def test_silence_does_not_produce_infinite_db(self, sample_rate):
        # The +1e-12 epsilon in the dB conversion should keep log10 finite.
        digital_zero = np.zeros(sample_rate, dtype=np.float32)
        result = ss.estimate_noise(digital_zero, sample_rate)
        assert np.isfinite(result["rms_db"])
        assert np.isfinite(result["noise_floor_db"])


class TestConfiguration:
    def test_default_config_is_not_mutated_by_overrides(self, clean_audio, sample_rate):
        before = copy.deepcopy(ss.DEFAULT_CONFIG)
        ss.detect_clicks(clean_audio, sample_rate, {"max_clicks_rate": 999.0})
        ss.detect_silence(clean_audio, sample_rate, {"silence_threshold_db": -10.0})
        assert ss.DEFAULT_CONFIG == before

    def test_partial_config_falls_back_to_defaults(self, clean_audio, sample_rate):
        # Only one key is supplied; the framing keys must still resolve.
        result = ss.detect_clicks(clean_audio, sample_rate, {"max_clicks_rate": 0.0})
        assert result["is_clicky"] is True


class TestAnalyzeRecording:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            ss.analyze_recording("does/not/exist.wav")

    def test_clean_file_reports_clean(self, clean_audio, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio, "clean.wav"))
        assert report["status"] == "CLEAN"
        assert report["issues"] == []
        assert report["file_name"] == "clean.wav"
        assert report["duration"] == pytest.approx(2.0, abs=0.05)

    def test_report_exposes_every_check(self, clean_audio, write_wav):
        report = ss.analyze_recording(write_wav(clean_audio))
        assert set(report) == {
            "file_name",
            "duration",
            "clicks",
            "saturation",
            "silence",
            "noise",
            "status",
            "issues",
        }

    def test_silent_file_reports_silent(self, clean_audio, write_wav):
        path = write_wav(ss.make_silent(clean_audio, level_db=-90), "silent.wav")
        report = ss.analyze_recording(path)
        assert report["status"] == "SILENT"
        assert any("silent" in issue.lower() for issue in report["issues"])

    def test_clipped_file_reports_saturated(self, clean_audio, write_wav):
        path = write_wav(ss.add_clipping(clean_audio, clip_threshold=0.01), "clipped.wav")
        report = ss.analyze_recording(path)
        assert report["status"] == "SATURATED"
        assert report["saturation"]["is_saturated"] is True

    def test_noisy_file_reports_noisy(self, clean_audio, write_wav):
        path = write_wav(ss.add_noise(clean_audio, target_snr_db=0), "noisy.wav")
        report = ss.analyze_recording(path)
        assert report["status"] == "NOISY"
        assert report["noise"]["is_noisy"] is True

    def test_silence_outranks_every_other_flag(self, clean_audio, write_wav):
        # Thresholds tuned so all four checks fail at once; silence must win.
        report = ss.analyze_recording(
            write_wav(clean_audio),
            {
                "min_silence_ratio": -1.0,
                "max_saturation_ratio": -1.0,
                "max_clicks_rate": -1.0,
                "min_snr_db": 1e6,
            },
        )
        assert len(report["issues"]) == 4
        assert report["status"] == "SILENT"

    def test_saturation_outranks_clicks_and_noise(self, clean_audio, write_wav):
        report = ss.analyze_recording(
            write_wav(clean_audio),
            {"max_saturation_ratio": -1.0, "max_clicks_rate": -1.0, "min_snr_db": 1e6},
        )
        assert len(report["issues"]) == 3
        assert report["status"] == "SATURATED"

    def test_clicks_outrank_noise(self, clean_audio, write_wav):
        report = ss.analyze_recording(
            write_wav(clean_audio), {"max_clicks_rate": -1.0, "min_snr_db": 1e6}
        )
        assert len(report["issues"]) == 2
        assert report["status"] == "CLICKY"

    def test_config_overrides_reach_the_individual_checks(self, clean_audio, write_wav):
        path = write_wav(clean_audio, "clean.wav")
        report = ss.analyze_recording(path, {"min_snr_db": 1e6})
        assert report["status"] == "NOISY"
