"""
Rebuild audio_examples/samples/ from the full field corpus.

The complete recordings run to roughly 900 MB and are not tracked in git. This
script distills them into a handful of short excerpts small enough to commit, so
that a fresh clone can still run the notebooks and the CLI against real audio.

Two properties matter and both are enforced below:

1. **Lossless.** The excerpts are FLAC. The `wind/` and `clicky/` material is
   defined by sample-level peak behaviour, and any lossy codec would round off
   the flat clipped tops that `detect_saturation` keys on.

2. **The label survives the cut.** Each window is chosen by the same measurement
   its folder is named after, and the resulting file is re-analyzed from disk to
   confirm it still reports the status the full recording does. A `wind/` sample
   that happened to land on a calm stretch would be worse than no sample at all.

Usage (from the repository root, with the full corpus present):

    uv run python scripts/make_samples.py
"""

from pathlib import Path

import essentia.standard as es
import numpy as np

import soundsanity as ss

SR = 48000                      # every source recording is 48 kHz mono 16-bit
EXCERPT_S = 10.0
HOP_S = 2.0                     # spacing between candidate windows
EDGE_S = 5.0                    # skipped at each end: recorders settle on startup

ROOT = Path(__file__).resolve().parent.parent / "audio_examples"
OUT = ROOT / "samples"

# Two per category. Where a category has only two recordings, both are used.
SOURCES = {
    "clean":   ["20230211_110222.WAV", "20230212_180055.WAV"],
    "clicky":  ["20230211_150000.WAV", "20230211_180000.WAV"],
    "noisy":   ["20230211_190000.WAV", "ruido01.wav"],
    "silence": ["20230211_050000.WAV", "20230212_180002.WAV"],
    "wind":    ["20231214_052000.WAV", "20231219_050000.WAV"],
}


def candidate_starts(n_samples):
    """Start offsets of every window we are willing to cut, in samples."""
    length = int(EXCERPT_S * SR)
    edge = int(EDGE_S * SR)
    last = n_samples - length - edge
    if last <= edge:                        # too short to trim: centre it
        return [max(0, (n_samples - length) // 2)]
    return list(range(edge, last, int(HOP_S * SR)))


def pick_window(audio, group):
    """
    Return (start_sample, reason) for the excerpt that best represents `group`.

    Scoring is per category, always by the property the folder is named for:
    the clickiest window of a clicky file, the most clipped window of a windy
    one, the quietest window of a silent one.
    """
    length = int(EXCERPT_S * SR)
    starts = candidate_starts(len(audio))

    if group == "clicky":
        times = ss.detect_clicks(audio, SR)["click_timestamps"]
        # ClickDetector can report a timestamp just past the final frame.
        idx = np.array([int(t * SR) for t in times if int(t * SR) < len(audio)])
        counts = [int(((idx >= s) & (idx < s + length)).sum()) for s in starts]
        best = int(np.argmax(counts))
        return starts[best], f"{counts[best]} clicks"

    if group == "wind":
        clipped = np.cumsum((np.abs(audio) > 0.99).astype(np.float64))
        clipped = np.concatenate(([0.0], clipped))
        ratios = [(clipped[s + length] - clipped[s]) / length for s in starts]
        best = int(np.argmax(ratios))
        return starts[best], f"{ratios[best]:.1%} clipped"

    if group == "silence":
        squared = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
        levels = np.array([
            np.sqrt((squared[s + length] - squared[s]) / length) for s in starts
        ])
        best = int(np.argmin(levels))
        return starts[best], f"RMS {20 * np.log10(levels[best] + 1e-12):.1f} dB"

    # clean and noisy are opposite ends of one test, and that test has two arms:
    # `is_noisy` trips on a high absolute noise floor OR a low SNR. Score each
    # window by its headroom in whichever arm is closer to tripping, so a window
    # cannot look clean on SNR alone while sitting above the noise-floor ceiling.
    cfg = ss.DEFAULT_CONFIG
    headroom = np.array([
        min(cfg["max_noise_floor_db"] - noise["noise_floor_db"],
            noise["snr_db"] - cfg["min_snr_db"])
        for noise in (
            ss.estimate_noise(audio[s:s + length], SR) for s in starts
        )
    ])
    best = int(np.argmax(headroom) if group == "clean" else np.argmin(headroom))
    return starts[best], f"headroom {headroom[best]:+.1f} dB"


def main():
    if not ROOT.exists():
        raise SystemExit(f"no corpus at {ROOT}")

    OUT.mkdir(parents=True, exist_ok=True)
    header = (f"{'group':<8} {'source':<26} {'at':>6}  {'chosen for':<18} "
              f"{'full':<10} {'excerpt':<10} {'KB':>6}")
    print(header)
    print("-" * len(header))

    total_kb = 0.0
    mismatched = []
    for group, names in SOURCES.items():
        (OUT / group).mkdir(exist_ok=True)
        for name in names:
            source = ROOT / group / name
            if not source.exists():
                raise SystemExit(f"missing source recording: {source}")

            audio = es.MonoLoader(filename=str(source), sampleRate=SR)()
            start, reason = pick_window(audio, group)
            excerpt = audio[start:start + int(EXCERPT_S * SR)]

            destination = OUT / group / f"{Path(name).stem}.flac"
            es.MonoWriter(
                filename=str(destination), format="flac", sampleRate=SR
            )(excerpt)

            # Judge the written file, not the array in memory, so this check
            # sees exactly what someone cloning the repo would load.
            full_status = ss.analyze_recording(str(source))["status"]
            excerpt_status = ss.analyze_recording(str(destination))["status"]
            size_kb = destination.stat().st_size / 1024
            total_kb += size_kb

            if full_status != excerpt_status:
                mismatched.append((destination, full_status, excerpt_status))

            print(f"{group:<8} {name:<26} {start / SR:5.0f}s  {reason:<18} "
                  f"{full_status:<10} {excerpt_status:<10} {size_kb:6.0f}"
                  f"{'' if full_status == excerpt_status else '   <-- DIFFERS'}")

    print(f"\nwrote {OUT.relative_to(ROOT.parent)} - {total_kb / 1024:.1f} MB")
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} excerpt(s) no longer report their source status; "
            "widen EXCERPT_S or choose a different source recording"
        )
    print("every excerpt still reports the status of the recording it came from")


if __name__ == "__main__":
    main()
