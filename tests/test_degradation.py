"""Tests for the simulated-defect helpers in soundsanity.degradation."""

import numpy as np
import pytest

import soundsanity as ss


class TestAddClicks:
    def test_length_is_preserved(self, clean_audio):
        degraded, _ = ss.add_clicks(clean_audio, num_clicks=20)
        assert degraded.shape == clean_audio.shape

    def test_returns_one_timestamp_per_click(self, clean_audio, sample_rate):
        degraded, timestamps = ss.add_clicks(
            clean_audio, num_clicks=20, sample_rate=sample_rate
        )
        assert len(timestamps) == 20
        assert timestamps == sorted(timestamps)
        duration = len(clean_audio) / sample_rate
        assert all(0.0 <= t <= duration for t in timestamps)

    def test_original_signal_is_not_mutated(self, clean_audio):
        original = clean_audio.copy()
        ss.add_clicks(clean_audio, num_clicks=20)
        np.testing.assert_array_equal(clean_audio, original)

    def test_clicks_land_at_the_reported_timestamps(self, clean_audio, sample_rate):
        degraded, timestamps = ss.add_clicks(
            clean_audio, num_clicks=10, click_amplitude=0.8, sample_rate=sample_rate
        )
        for t in timestamps:
            idx = int(round(t * sample_rate))
            assert degraded[idx] == pytest.approx(0.8, abs=1e-6)

    def test_very_short_audio_is_returned_untouched(self):
        short = np.zeros(20, dtype=np.float32)
        degraded, timestamps = ss.add_clicks(short, num_clicks=5)
        assert timestamps == []
        np.testing.assert_array_equal(degraded, short)


class TestAddClipping:
    def test_output_stays_within_full_scale(self, clean_audio):
        clipped = ss.add_clipping(clean_audio, clip_threshold=0.01)
        assert np.max(np.abs(clipped)) <= 1.0 + 1e-6

    def test_signal_is_flat_topped(self, clean_audio):
        clipped = ss.add_clipping(clean_audio, clip_threshold=0.01)
        # Aggressive clipping should pin a large share of samples to the rails.
        at_rails = np.mean(np.isclose(np.abs(clipped), 1.0, atol=1e-6))
        assert at_rails > 0.5

    def test_higher_threshold_clips_less(self, clean_audio):
        hard = ss.add_clipping(clean_audio, clip_threshold=0.01)
        soft = ss.add_clipping(clean_audio, clip_threshold=0.9)
        rails = lambda x: np.mean(np.isclose(np.abs(x), 1.0, atol=1e-6))  # noqa: E731
        assert rails(soft) < rails(hard)

    def test_all_zero_input_does_not_divide_by_zero(self):
        silence = np.zeros(1000, dtype=np.float32)
        clipped = ss.add_clipping(silence, clip_threshold=0.01)
        assert np.all(np.isfinite(clipped))
        np.testing.assert_allclose(clipped, 0.0)


class TestAddNoise:
    def test_shape_is_preserved(self, clean_audio):
        assert ss.add_noise(clean_audio, target_snr_db=10).shape == clean_audio.shape

    def test_lower_target_snr_adds_more_noise(self, clean_audio):
        quiet = ss.add_noise(clean_audio, target_snr_db=30)
        loud = ss.add_noise(clean_audio, target_snr_db=0)
        residual = lambda x: np.std(x - clean_audio)  # noqa: E731
        assert residual(loud) > residual(quiet)

    def test_output_is_normalized_below_full_scale(self, clean_audio):
        noisy = ss.add_noise(clean_audio, target_snr_db=-10)
        assert np.max(np.abs(noisy)) <= 1.0 + 1e-6

    def test_achieved_snr_tracks_the_target(self, clean_audio):
        target_db = 20.0
        noisy = ss.add_noise(clean_audio, target_snr_db=target_db)
        noise = noisy - clean_audio
        achieved = 10 * np.log10(np.mean(clean_audio**2) / np.mean(noise**2))
        assert achieved == pytest.approx(target_db, abs=1.0)

    def test_silent_input_does_not_divide_by_zero(self):
        silence = np.zeros(1000, dtype=np.float32)
        noisy = ss.add_noise(silence, target_snr_db=10)
        assert np.all(np.isfinite(noisy))


class TestMakeSilent:
    def test_attenuates_by_the_requested_amount(self, clean_audio):
        attenuated = ss.make_silent(clean_audio, level_db=-40)
        rms = lambda x: np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))  # noqa: E731
        drop_db = 20 * np.log10(rms(attenuated) / rms(clean_audio))
        assert drop_db == pytest.approx(-40.0, abs=0.1)

    def test_zero_db_leaves_the_signal_unchanged(self, clean_audio):
        np.testing.assert_allclose(ss.make_silent(clean_audio, level_db=0), clean_audio)

    def test_shape_is_preserved(self, clean_audio):
        assert ss.make_silent(clean_audio).shape == clean_audio.shape
