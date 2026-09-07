"""Tests for the combined per-recording field analysis."""

import numpy as np
import pytest

import essentia.standard as es

from soundsanity.field import (
    FIELD_FIELDS,
    analyze_field_recording,
    analyze_field_task,
    band_column_names,
    load_window,
)

SAMPLE_RATE = 48000


@pytest.fixture
def field_wav(tmp_path):
    """Write a WAV at the corpus's native rate."""

    def _write(audio, name="20231210_195000.wav"):
        path = tmp_path / name
        es.MonoWriter(
            filename=str(path), format="wav", sampleRate=SAMPLE_RATE
        )(np.asarray(audio, dtype=np.float32))
        return str(path)

    return _write


@pytest.fixture
def long_recording():
    """Ten seconds whose two halves differ, so a window can be told from the whole."""
    time = np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE
    audio = 0.2 * np.sin(2 * np.pi * 100.0 * time)
    audio[SAMPLE_RATE * 5:] *= 0.01
    return audio.astype(np.float32)


class TestBandColumnNames:
    def test_names_one_column_per_band(self):
        names = band_column_names([20.0, 50.0, 100.0])
        assert names == ["band_20_50_db", "band_50_100_db"]


class TestLoadWindow:
    def test_reads_only_the_requested_window(self, field_wav, long_recording):
        path = field_wav(long_recording)
        assert len(load_window(path, window_s=2.0)) == pytest.approx(
            SAMPLE_RATE * 2, rel=0.01
        )

    def test_offset_skips_into_the_file(self, field_wav, long_recording):
        path = field_wav(long_recording)
        head = load_window(path, window_s=1.0)
        tail = load_window(path, window_s=1.0, offset_s=6.0)
        # The back half was attenuated, so the tail must be far quieter.
        assert np.abs(tail).max() < np.abs(head).max() / 10

    def test_window_none_loads_the_whole_file(self, field_wav, long_recording):
        path = field_wav(long_recording)
        assert len(load_window(path, window_s=None)) == pytest.approx(
            SAMPLE_RATE * 10, rel=0.01
        )

    def test_window_longer_than_the_file_returns_what_exists(self, field_wav, long_recording):
        path = field_wav(long_recording)
        assert len(load_window(path, window_s=60.0)) == pytest.approx(
            SAMPLE_RATE * 10, rel=0.01
        )

    def test_levels_are_absolute(self, field_wav):
        # EasyLoader applies replayGain; the field pipeline needs unity, or every
        # level it reports would be shifted and the weather join meaningless.
        constant = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
        loaded = load_window(field_wav(constant), window_s=1.0)
        assert np.abs(loaded).max() == pytest.approx(0.5, abs=1e-3)


class TestAnalyzeFieldRecording:
    def test_returns_every_declared_column(self, field_wav, long_recording):
        row = analyze_field_recording(field_wav(long_recording), window_s=2.0)
        assert set(FIELD_FIELDS) <= set(row)

    def test_includes_a_column_per_band(self, field_wav, long_recording):
        row = analyze_field_recording(field_wav(long_recording), window_s=2.0)
        assert "band_20_50_db" in row
        assert "band_16000_24000_db" in row

    def test_reports_the_analyzed_duration_not_the_file_duration(
        self, field_wav, long_recording
    ):
        row = analyze_field_recording(field_wav(long_recording), window_s=3.0)
        assert row["analyzed_s"] == pytest.approx(3.0, rel=0.01)

    def test_carries_both_quality_and_wind_results(self, field_wav, long_recording):
        row = analyze_field_recording(field_wav(long_recording), window_s=2.0)
        assert row["status"] in {"CLEAN", "NOISY", "CLICKY", "SATURATED", "SILENT"}
        assert row["lf_ratio"] is not None
        assert row["noise_floor_db"] is not None

    def test_low_frequency_recording_is_flagged_windy(self, field_wav):
        time = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
        rumble = (0.4 * np.sin(2 * np.pi * 80.0 * time)).astype(np.float32)
        row = analyze_field_recording(field_wav(rumble), window_s=2.0)
        assert row["is_windy"]

    def test_missing_file_returns_an_error_row_rather_than_raising(self, tmp_path):
        row = analyze_field_recording(str(tmp_path / "gone.wav"))
        assert row["status"] == "ERROR"
        assert row["error"]
        assert row["lf_ratio"] is None

    def test_unreadable_file_returns_an_error_row(self, tmp_path):
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"not audio")
        row = analyze_field_recording(str(broken))
        assert row["status"] == "ERROR"

    def test_error_row_still_names_the_file(self, tmp_path):
        row = analyze_field_recording(str(tmp_path / "20231210_195000.wav"))
        assert row["file_name"] == "20231210_195000.wav"

    def test_config_overrides_reach_both_analyses(self, field_wav, long_recording):
        path = field_wav(long_recording)
        strict = analyze_field_recording(
            path, wind_config={"min_wind_index": 1.1}, window_s=2.0
        )
        assert not strict["is_windy"]


class TestAnalyzeFieldTask:
    def test_unpacks_a_worker_task(self, field_wav, long_recording):
        path = field_wav(long_recording)
        row = analyze_field_task((path, {"window_s": 2.0}))
        assert row["path"] == path
        assert row["analyzed_s"] == pytest.approx(2.0, rel=0.01)
