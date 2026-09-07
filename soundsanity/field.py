"""
Per-recording analysis for the field corpus: quality checks plus wind features.
"""

from __future__ import annotations

import os

import essentia.standard as es

from .analysis import DEFAULT_CONFIG, analyze_audio
from .wind import WIND_CONFIG, analyze_wind

#: Native rate of the field corpus. Loading at the recording's own rate avoids
#: a resample that would cost time and blur the band edges.
FIELD_SAMPLE_RATE = 48000

#: Seconds analyzed per recording. One minute is the shortest file in the
#: corpus, so it is the longest window every season can supply.
DEFAULT_WINDOW_S = 60.0

#: Columns `analyze_field_recording` returns, before the per-band levels.
FIELD_FIELDS = (
    "path",
    "file_name",
    "analyzed_s",
    "status",
    "issues",
    "rms_db",
    "noise_floor_db",
    "active_level_db",
    "snr_db",
    "silence_ratio",
    "is_silent",
    "clicks_rate",
    "clicks_count",
    "is_clicky",
    "saturation_ratio",
    "is_saturated",
    "lf_ratio",
    "lf_ratio_p90",
    "lf_level_db",
    "hf_level_db",
    "spectral_tilt",
    "wind_index",
    "is_windy",
    "error",
)


def band_column_names(edges):
    """
    Name one column per frequency band.

    Args:
        edges (sequence of float): Band edges in Hz.

    Returns:
        list of str: Names of the form ``band_20_50_db``.
    """
    return [
        f"band_{int(low)}_{int(high)}_db"
        for low, high in zip(edges[:-1], edges[1:])
    ]


def load_window(path, window_s=DEFAULT_WINDOW_S, offset_s=0.0,
                sample_rate=FIELD_SAMPLE_RATE):
    """
    Load a fixed-length window from a recording.

    Args:
        path (str): Audio file to read.
        window_s (float): Seconds to load. ``None`` or 0 loads the whole file.
        offset_s (float): Seconds to skip at the start.
        sample_rate (int): Rate to load at.

    Returns:
        numpy.ndarray: Mono samples. Shorter than `window_s` if the file is.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    if not window_s:
        return es.MonoLoader(filename=path, sampleRate=sample_rate)()

    # EasyLoader's replayGain defaults to -6 dB, which it treats as unity; it is
    # passed explicitly so that levels stay absolute if that default ever moves.
    return es.EasyLoader(
        filename=path,
        sampleRate=sample_rate,
        startTime=offset_s,
        endTime=offset_s + window_s,
        replayGain=-6.0,
    )()


def analyze_field_recording(path, config=None, wind_config=None,
                            window_s=DEFAULT_WINDOW_S, offset_s=0.0,
                            sample_rate=FIELD_SAMPLE_RATE):
    """
    Run the quality checks and the wind features over one recording.

    Args:
        path (str): Audio file to analyze.
        config (dict, optional): Overrides merged into `DEFAULT_CONFIG`.
        wind_config (dict, optional): Overrides merged into `WIND_CONFIG`.
        window_s (float): Seconds to analyze, from `offset_s`.
        offset_s (float): Seconds to skip at the start.
        sample_rate (int): Rate to load at.

    Returns:
        dict: One flat row — every name in FIELD_FIELDS, plus one
        ``band_<low>_<high>_db`` column per frequency band. A file that fails to
        load returns a row with ``error`` set and the metrics left as None,
        because at corpus scale a single bad file must not stop the run.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {}), "sample_rate": sample_rate}
    wcfg = {**WIND_CONFIG, **(wind_config or {})}

    row = {field: None for field in FIELD_FIELDS}
    row["path"] = str(path)
    row["file_name"] = os.path.basename(path)
    row["error"] = ""

    try:
        audio = load_window(str(path), window_s, offset_s, sample_rate)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "ERROR"
        return row

    try:
        quality = analyze_audio(audio, sample_rate, cfg)
        wind = analyze_wind(audio, sample_rate, wcfg)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "ERROR"
        return row

    row.update(
        {
            "analyzed_s": quality["duration"],
            "status": quality["status"],
            "issues": "; ".join(quality["issues"]),
            "rms_db": quality["noise"]["rms_db"],
            "noise_floor_db": quality["noise"]["noise_floor_db"],
            "active_level_db": quality["noise"]["active_level_db"],
            "snr_db": quality["noise"]["snr_db"],
            "silence_ratio": quality["silence"]["silence_ratio"],
            "is_silent": quality["silence"]["is_silent"],
            "clicks_rate": quality["clicks"]["clicks_rate"],
            "clicks_count": quality["clicks"]["clicks_count"],
            "is_clicky": quality["clicks"]["is_clicky"],
            "saturation_ratio": quality["saturation"]["saturation_ratio"],
            "is_saturated": quality["saturation"]["is_saturated"],
            "lf_ratio": wind["lf_ratio"],
            "lf_ratio_p90": wind["lf_ratio_p90"],
            "lf_level_db": wind["lf_level_db"],
            "hf_level_db": wind["hf_level_db"],
            "spectral_tilt": wind["spectral_tilt"],
            "wind_index": wind["wind_index"],
            "is_windy": wind["is_windy"],
        }
    )

    for name, level in zip(
        band_column_names(wind["band_edges_hz"]), wind["band_levels_db"]
    ):
        row[name] = level

    return row


def analyze_field_task(task):
    """
    Worker entry point for `ProcessPoolExecutor`.

    Must stay module-level and take a single picklable argument.

    Args:
        task (tuple): ``(path, kwargs)`` for `analyze_field_recording`.

    Returns:
        dict: The row `analyze_field_recording` produced.
    """
    path, kwargs = task
    return analyze_field_recording(path, **kwargs)
