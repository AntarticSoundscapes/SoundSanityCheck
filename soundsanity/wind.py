"""
Wind-noise features for field recordings.
"""

from __future__ import annotations

import numpy as np
import essentia.standard as es

#: Band edges in Hz. The lowest starts at 20 Hz rather than 0, below anything
#: the recorder resolves. The top edge is clamped to Nyquist at analysis time.
DEFAULT_BAND_EDGES = (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0,
                       4000.0, 8000.0, 16000.0, 24000.0)

WIND_CONFIG = {
    "band_edges_hz": DEFAULT_BAND_EDGES,
    "wind_band_hz": 200.0,      # Upper edge of the band wind dominates
    "frame_size": 2048,         # Longer than the default framing: 1024 samples
    "hop_size": 1024,           # at 48 kHz gives 47 Hz bins, too coarse below 200 Hz

    # Fitted against the station anemometer; see the module docstring. Levels are
    # raw dBFS, so a deployment recorded at a different gain needs them refitted.
    "min_wind_index": 0.47,     # Above this a recording is called windy
    "wind_lf_level_db": -30.0,  # Low band level at which the level term starts
    "wind_level_span_db": 25.0, # ...and over how many dB it reaches full weight
    "lf_ratio_floor": 0.35,     # Share of energy the low band must also hold
}


def wind_index_from_features(lf_ratio, lf_level_db, config=None):
    """
    Compute the wind index and flag from the two features it depends on.

    Factored out so that a recalibration can be applied to an existing analysis
    without decoding audio again: `lf_ratio` and `lf_level_db` are both stored
    per recording, and the index is a pure function of them. Accepts scalars or
    arrays.

    Args:
        lf_ratio (float | numpy.ndarray): Share of energy below the wind band.
        lf_level_db (float | numpy.ndarray): Level of that band, in dBFS.
        config (dict, optional): Overrides merged into WIND_CONFIG.

    Returns:
        tuple: ``(wind_index, is_windy)``, matching the shape of the inputs.
    """
    cfg = {**WIND_CONFIG, **(config or {})}

    level_term = np.clip(
        (np.asarray(lf_level_db, dtype=float) - cfg["wind_lf_level_db"])
        / cfg["wind_level_span_db"],
        0.0,
        1.0,
    )
    index = np.asarray(lf_ratio, dtype=float) * level_term
    is_windy = (index >= cfg["min_wind_index"]) & (
        np.asarray(lf_ratio, dtype=float) >= cfg["lf_ratio_floor"]
    )
    return index, is_windy


def _band_levels(audio, sample_rate, cfg):
    """
    Compute per-frame energy in each band.

    Returns:
        tuple of (np.ndarray, np.ndarray): A (frames, bands) energy matrix and
        the band edges actually used, clamped to Nyquist.
    """
    nyquist = sample_rate / 2.0
    edges = [e for e in cfg["band_edges_hz"] if e < nyquist]
    edges.append(nyquist)

    # Remove DC before any spectrum is taken. AudioMoth recordings routinely
    # carry a standing offset, and starting the lowest band at 20 Hz is not
    # enough to keep it out: a Hann window spreads DC across the first few bins,
    # so an offset of 0.5 lands in the 20-50 Hz band at -6 dB and reads as a
    # gale. Subtracting the mean of the whole signal cancels it exactly, and
    # over a window of many seconds it removes nothing else — the implied
    # cutoff is far below the lowest band edge.
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - audio.mean()

    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    bands = es.FrequencyBands(frequencyBands=edges, sampleRate=sample_rate)

    rows = []
    for frame in es.FrameGenerator(
        audio, frameSize=cfg["frame_size"], hopSize=cfg["hop_size"]
    ):
        rows.append(bands(spectrum(window(frame))))

    if not rows:
        return np.empty((0, len(edges) - 1)), np.array(edges)
    return np.vstack(rows), np.array(edges)


def analyze_wind(audio, sample_rate, config=None):
    """
    Measure the wind signature of an audio signal.

    Levels are reported as the median across frames rather than the mean: a
    single boat pass or a bird call directly on the microphone would drag a mean
    upward, while wind — which is what this function is for — is sustained and
    sits at the median.

    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Overrides merged into WIND_CONFIG.

    Returns:
        dict: Wind metrics:
            - lf_ratio (float): Share of energy below ``wind_band_hz``, 0-1.
            - lf_level_db (float): Median level of that low band, in dBFS.
            - hf_level_db (float): Median level above the low band, in dBFS.
            - spectral_tilt (float): dB per decade across band centres; steeply
              negative for wind.
            - wind_index (float): Combined 0-1 score.
            - is_windy (bool): Whether the score and level both clear threshold.
            - lf_ratio_p90 (float): 90th-percentile frame `lf_ratio`, which
              catches recordings that are calm apart from a few strong gusts.
            - band_levels_db (list of float): Median level per band.
            - band_edges_hz (list of float): The band edges used.
    """
    cfg = {**WIND_CONFIG, **(config or {})}

    empty = {
        "lf_ratio": 0.0,
        "lf_level_db": -100.0,
        "hf_level_db": -100.0,
        "spectral_tilt": 0.0,
        "wind_index": 0.0,
        "is_windy": False,
        "lf_ratio_p90": 0.0,
        "band_levels_db": [],
        "band_edges_hz": [],
    }
    if len(audio) == 0:
        return empty

    energy, edges = _band_levels(audio, sample_rate, cfg)
    if energy.shape[0] == 0:
        return empty

    # Split the bands at the wind cutoff. `edges[1:]` are the upper edges, so a
    # band counts as low when its upper edge is at or below the cutoff.
    is_low = edges[1:] <= cfg["wind_band_hz"]

    total = energy.sum(axis=1)
    low = energy[:, is_low].sum(axis=1)
    high = energy[:, ~is_low].sum(axis=1)

    # Frames with no energy at all (digital silence) would make the ratio 0/0.
    valid = total > 0
    frame_ratio = np.zeros_like(total)
    frame_ratio[valid] = low[valid] / total[valid]

    band_levels_db = 10 * np.log10(np.median(energy, axis=0) + 1e-20)
    centres = np.sqrt(edges[:-1] * edges[1:])

    # Tilt: least-squares slope of band level against log10(centre frequency).
    # Bands that are effectively empty are dropped so the floor does not flatten
    # the fit.
    audible = band_levels_db > -110.0
    if audible.sum() >= 2:
        slope = np.polyfit(np.log10(centres[audible]), band_levels_db[audible], 1)[0]
    else:
        slope = 0.0

    lf_ratio = float(np.median(frame_ratio))
    lf_level_db = float(10 * np.log10(np.median(low) + 1e-20))
    hf_level_db = float(10 * np.log10(np.median(high) + 1e-20))

    # The index rewards a low-frequency-dominated spectrum, but only once the
    # low band is actually loud: a quiet recording with a gentle roll-off would
    # otherwise score as a gale. The level term saturates `wind_level_span_db`
    # above the floor, so a storm and a stronger storm both read as fully windy
    # and the index stays bounded. Both figures are fitted — placed too low they
    # saturate over the whole corpus and the index stops discriminating at all.
    index, windy = wind_index_from_features(lf_ratio, lf_level_db, cfg)
    wind_index = float(index)
    is_windy = bool(windy)

    return {
        "lf_ratio": lf_ratio,
        "lf_level_db": lf_level_db,
        "hf_level_db": hf_level_db,
        "spectral_tilt": float(slope),
        "wind_index": wind_index,
        "is_windy": is_windy,
        "lf_ratio_p90": float(np.percentile(frame_ratio, 90)),
        "band_levels_db": [float(v) for v in band_levels_db],
        "band_edges_hz": [float(v) for v in edges],
    }
