"""Shared fixtures for the soundsanity test suite."""

import matplotlib
import numpy as np
import pytest

# Render plots off-screen so the plotting tests run headless (e.g. in CI).
matplotlib.use("Agg")

import essentia.standard as es  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SAMPLE_RATE = 44100


@pytest.fixture
def sample_rate():
    """Sample rate used throughout the suite (matches analyze_recording's loader)."""
    return SAMPLE_RATE


@pytest.fixture(autouse=True)
def seeded_rng():
    """Make the random-based degradation helpers deterministic across runs."""
    np.random.seed(20230211)


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test opened so matplotlib doesn't leak state."""
    yield
    plt.close("all")


@pytest.fixture
def clean_audio(sample_rate):
    """
    A 2-second synthetic recording that passes every quality check.

    A 440 Hz tone sits at a near-silent floor for the first 0.7 s and then ramps
    up to full level, which gives the frame-RMS distribution enough spread for a
    healthy SNR without tripping the silence, click, or saturation detectors.
    """
    duration = 2.0
    n_samples = int(sample_rate * duration)
    time = np.arange(n_samples) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * time)

    floor = 1e-4
    envelope = np.full(n_samples, floor)
    onset = int(0.7 * sample_rate)
    ramp_len = int(0.1 * sample_rate)
    # Raised-cosine ramp: an abrupt onset would read as a click.
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
    envelope[onset:onset + ramp_len] = floor + (1.0 - floor) * ramp
    envelope[onset + ramp_len:] = 1.0

    return (tone * envelope).astype(np.float32)


@pytest.fixture
def empty_audio():
    """A zero-length signal, for exercising the guard clauses."""
    return np.array([], dtype=np.float32)


@pytest.fixture
def write_wav(tmp_path, sample_rate):
    """Return a helper that writes a signal to a temporary WAV file."""

    def _write(audio, name="test.wav"):
        path = tmp_path / name
        es.MonoWriter(filename=str(path), format="wav", sampleRate=sample_rate)(
            np.asarray(audio, dtype=np.float32)
        )
        return str(path)

    return _write
