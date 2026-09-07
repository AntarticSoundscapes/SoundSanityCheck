"""Tests for the wind-noise features."""

import numpy as np
import pytest

from soundsanity.wind import WIND_CONFIG, analyze_wind, wind_index_from_features

SAMPLE_RATE = 48000


def _signal(freq, seconds=2.0, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """A steady tone, as a stand-in for energy concentrated at one frequency."""
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


@pytest.fixture
def windy_audio():
    """
    Low-frequency-dominated noise, the shape wind makes on a microphone.

    Built by low-pass filtering white noise with a running mean, which rolls the
    spectrum off steeply above ~100 Hz.
    """
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(SAMPLE_RATE * 2)
    kernel = np.ones(240) / 240  # ~200 Hz cutoff at 48 kHz
    smoothed = np.convolve(noise, kernel, mode="same")
    return (0.5 * smoothed / np.abs(smoothed).max()).astype(np.float32)


@pytest.fixture
def calm_audio():
    """A quiet, high-frequency signal: loud enough to measure, not wind-shaped."""
    return _signal(4000.0, amplitude=0.3)


class TestAnalyzeWind:
    def test_low_frequency_noise_reads_as_windy(self, windy_audio):
        result = analyze_wind(windy_audio, SAMPLE_RATE)
        assert result["is_windy"]
        assert result["lf_ratio"] > 0.8
        assert result["spectral_tilt"] < 0

    def test_high_frequency_signal_does_not(self, calm_audio):
        result = analyze_wind(calm_audio, SAMPLE_RATE)
        assert not result["is_windy"]
        assert result["lf_ratio"] < 0.1

    def test_the_level_ramp_is_configurable(self):
        # The fitted floor and span are corpus-specific; a deployment at another
        # gain must be able to move them.
        faint = _signal(60.0, amplitude=0.001)
        assert analyze_wind(faint, SAMPLE_RATE)["wind_index"] == 0.0
        refitted = analyze_wind(
            faint, SAMPLE_RATE,
            {"wind_lf_level_db": -70.0, "wind_level_span_db": 20.0},
        )
        assert refitted["wind_index"] > 0.5

    def test_a_quiet_low_frequency_signal_is_not_called_wind(self):
        # Shape alone is not enough: without energy in the low band this is just
        # a quiet recording, and calling it a gale is the failure mode the level
        # term exists to prevent.
        faint = _signal(60.0, amplitude=1e-4)
        result = analyze_wind(faint, SAMPLE_RATE)
        assert result["lf_ratio"] > 0.9
        assert not result["is_windy"]

    def test_wind_index_rises_with_level(self):
        # Amplitudes chosen inside the level term's fitted ramp, which spans
        # -30 to -5 dBFS. Outside it the index saturates by design.
        indices = [
            analyze_wind(_signal(60.0, amplitude=a), SAMPLE_RATE)["wind_index"]
            for a in (0.03, 0.1, 0.3)
        ]
        assert indices[0] < indices[1] < indices[2]

    def test_wind_index_stays_bounded(self):
        result = analyze_wind(_signal(60.0, amplitude=0.99), SAMPLE_RATE)
        assert 0.0 <= result["wind_index"] <= 1.0

    def test_dc_offset_is_not_mistaken_for_wind(self):
        # AudioMoth recordings routinely carry a DC offset. The 20 Hz band edge
        # does not keep it out on its own — a Hann window leaks DC into the
        # first bins — so the signal mean is removed before the spectrum.
        offset = np.full(SAMPLE_RATE * 2, 0.5, dtype=np.float32)
        assert not analyze_wind(offset, SAMPLE_RATE)["is_windy"]

    def test_bands_are_clamped_to_nyquist(self):
        result = analyze_wind(_signal(1000.0, sample_rate=16000), 16000)
        assert result["band_edges_hz"][-1] == 8000.0
        assert len(result["band_levels_db"]) == len(result["band_edges_hz"]) - 1

    def test_p90_catches_a_recording_that_is_mostly_calm(self, windy_audio, calm_audio):
        # A gust in an otherwise quiet recording: the median stays low, so the
        # 90th percentile is what reports it.
        gusty = np.concatenate([calm_audio] * 3 + [windy_audio]).astype(np.float32)
        result = analyze_wind(gusty, SAMPLE_RATE)
        assert result["lf_ratio_p90"] > result["lf_ratio"]

    def test_empty_signal_returns_the_neutral_row(self):
        result = analyze_wind(np.array([], dtype=np.float32), SAMPLE_RATE)
        assert not result["is_windy"]
        assert result["wind_index"] == 0.0
        assert result["band_levels_db"] == []

    def test_signal_shorter_than_a_frame_returns_the_neutral_row(self):
        short = np.zeros(WIND_CONFIG["frame_size"] // 4, dtype=np.float32)
        assert analyze_wind(short, SAMPLE_RATE)["wind_index"] == 0.0

    def test_digital_silence_does_not_divide_by_zero(self):
        result = analyze_wind(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        assert result["lf_ratio"] == 0.0
        assert not result["is_windy"]

    def test_config_overrides_are_honored(self, windy_audio):
        strict = analyze_wind(windy_audio, SAMPLE_RATE, {"min_wind_index": 1.1})
        assert not strict["is_windy"]

    def test_wind_band_cutoff_shifts_the_ratio(self, calm_audio):
        # Widening the band to include 4 kHz should capture the tone.
        wide = analyze_wind(calm_audio, SAMPLE_RATE, {"wind_band_hz": 8000.0})
        assert wide["lf_ratio"] > 0.9


class TestWindIndexFromFeatures:
    """Recalibration must be applicable to a stored analysis without re-decoding."""

    def test_matches_what_analyze_wind_computes(self, windy_audio):
        full = analyze_wind(windy_audio, SAMPLE_RATE)
        index, windy = wind_index_from_features(full["lf_ratio"], full["lf_level_db"])
        assert float(index) == pytest.approx(full["wind_index"])
        assert bool(windy) == full["is_windy"]

    def test_accepts_arrays(self):
        index, windy = wind_index_from_features(
            np.array([0.9, 0.9, 0.1]), np.array([-5.0, -60.0, -5.0])
        )
        assert index.shape == (3,)
        # Loud and low-frequency; quiet; loud but not low-frequency.
        assert windy.tolist() == [True, False, False]

    def test_honors_a_refitted_threshold(self):
        _, windy = wind_index_from_features(0.9, -5.0, {"min_wind_index": 1.1})
        assert not bool(windy)

    def test_level_floor_moves_the_result(self):
        quiet = -55.0
        assert not bool(wind_index_from_features(0.9, quiet)[1])
        assert bool(
            wind_index_from_features(
                0.9, quiet, {"wind_lf_level_db": -70.0, "wind_level_span_db": 15.0}
            )[1]
        )

    def test_ratio_floor_still_gates(self):
        # A loud but broadband recording is not wind however high the level is.
        _, windy = wind_index_from_features(0.2, 10.0)
        assert not bool(windy)
