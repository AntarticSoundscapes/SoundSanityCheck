# Antarctic Soundscape Sanity Checker

This project provides a Python library (`soundsanity`) built on top of the **Essentia** audio analysis library to perform automated sanity checks and quality diagnostics on environmental field recordings from Antarctica. 

Field recordings can suffer from various technical issues during deployment:
- **Too many clicks**: Impulsive pops or clicks.
- **Clipped/Saturated audio**: Input gain set too high, leading to flat-topped waveform distortion.
- **Failed recordings (Silence)**: Microphone failure or possible failures in the recording device's software yielding empty channels.
- **Overly noisy files**: Excessive wind.

This library provides a set of diagnostic functions leveraging Essentia's `Audio Problems` algorithms, visualizes the issues, and offers tools to simulate these defects on clean audio to test the robust detection parameters.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Setting up the environment
From the project root:

```bash
uv sync
```

This creates a `.venv/`, installs every dependency at the exact versions recorded in
`uv.lock`, and installs `soundsanity` itself in editable mode.

### Common Tasks

| Task | Command |
| --- | --- |
| Sync the environment | `uv sync` |
| Add a dependency | `uv add <package>` |
| Add a development-only dependency | `uv add --dev <package>` |
| Remove a dependency | `uv remove <package>` |
| Upgrade all locked versions | `uv lock --upgrade` |
| Run a command in the environment | `uv run <command>` |

---

## Library Architecture

The project is structured as follows:

```text
SoundSanityCheck/
├── audio_examples/         # Example soundscape recordings (WAV/MP3)
│   └── samples/            # Committed 10-second excerpts, by category
├── notebooks/              # Jupyter Notebooks for exploration and demos
│   ├── 01_exploratory_analysis.ipynb  # Running checks on baseline clean files
│   ├── 02_sanity_check_demo.ipynb     # Simulating issues and validating detections
│   └── 03_batch_report.ipynb          # Analyzing a whole directory at once
├── soundsanity/            # Core library package
│   ├── __init__.py         # Package entrypoint (exposes public API)
│   ├── analysis.py         # Diagnostic checks (Clicks, Saturation, Silence, Noise)
│   ├── batch.py            # Directory-level analysis and summaries
│   ├── cli.py              # `soundsanity-report` command-line front end
│   ├── degradation.py      # Audio degradation utilities for simulated testing
│   └── plots.py            # Diagnostic plotting functions
├── tests/                  # pytest suite
│   ├── conftest.py         # Shared fixtures (synthetic signals, temp WAV writer)
│   ├── test_analysis.py    # Diagnostic checks and overall status logic
│   ├── test_batch.py       # Directory scanning, batch runs, summaries
│   ├── test_cli.py         # Table rendering and command-line behavior
│   ├── test_degradation.py # Degradation helpers
│   └── test_plots.py       # Plot structure smoke tests
├── scripts/
│   └── make_samples.py     # Rebuilds audio_examples/samples/ from the full corpus
├── README.md               # Project documentation
├── pyproject.toml          # Project metadata and dependencies
├── .python-version         # Python version pinned for uv
└── uv.lock                 # Fully resolved, reproducible dependency set
```

---

## Core Quality Checks

### 1. Click Detection (`detect_clicks`)
Uses Essentia's `ClickDetector` (based on LPC inverse filtering and prediction error thresholds) to count individual clicks and record their timestamps. A recording is flagged as `CLICKY` if the average click rate exceeds a threshold (e.g. 1.0 click/sec).

### 2. Saturation Detection (`detect_saturation`)
Uses Essentia's `SaturationDetector` (based on energy thresholds and minimum difference between contiguous samples) to identify flat-topped clipped intervals. It calculates the total duration and ratio of saturation, flagging the recording as `SATURATED` if it exceeds a threshold (e.g. 0.1% of total length).

### 3. Silence Detection (`detect_silence`)
Uses Essentia's `SilenceRate` with linear energy thresholds to detect silent frames. It computes the silence ratio and flags a recording as `SILENT` (failed recording) if it is silent for a high percentage of its duration (e.g. >95%).

### 4. Noise & SNR Estimation (`estimate_noise`)
Computes the overall RMS level in dB and analyzes the distribution of frame-wise RMS values. It calculates:
- **Noise Floor**: The 10th percentile of frame energies (robust to brief transient calls).
- **Active Signal Level**: The 90th percentile of frame energies.
- **SNR index**: Active level minus the noise floor.
A file is flagged as `NOISY` if the noise floor exceeds a threshold or the SNR is too low.

---

## Customizing Configuration

You can customize all detection thresholds by passing a configuration dictionary to the analysis functions. The default settings in `ss.DEFAULT_CONFIG` are:

```python
DEFAULT_CONFIG = {
    # Click detection config
    "click_detection_threshold": 30.0,
    "max_clicks_rate": 1.0,               # Clicks/sec above which audio is flagged as 'clicky'
    "click_merge_tolerance": 0.01,        # Seconds within which clicks are grouped as one event
    
    # Saturation/Clipping config
    "saturation_energy_threshold": -1.0,  # dB threshold for sample energy in saturated region
    "saturation_diff_threshold": 0.001,   # Minimum difference between consecutive saturated samples
    "saturation_min_duration": 1.0,       # Minimum duration of saturated region in ms
    "max_saturation_ratio": 0.001,        # Ratio of clipping above which audio is flagged as 'saturated' (0.1%)
    
    # Silence/Failed recording config
    "silence_threshold_db": -50.0,        # dB level below which a frame is considered silent
    "min_silence_ratio": 0.95,            # Ratio of silence above which audio is flagged as 'silent'
    
    # Noise/SNR estimation config
    "noise_floor_percentile": 10.0,       # Percentile of frame RMS values used as noise floor
    "active_signal_percentile": 90.0,     # Percentile of frame RMS values used as active signal
    "max_noise_floor_db": -45.0,          # Noise floor above this is flagged as 'noisy'
    "min_snr_db": 10.0,                   # SNR below this is flagged as 'noisy'
}
```

---

## Code Example

```python
import soundsanity as ss
import essentia.standard as es

# Load and inspect a file
file_path = "audio_examples/pinguinos1.wav"
report = ss.analyze_recording(file_path)

print(f"Status: {report['status']}")
if report['issues']:
    print("Issues found:")
    for issue in report['issues']:
        print(f"  - {issue}")
else:
    print("Recording is clean!")

# Load audio signal for plotting
audio = es.MonoLoader(filename=file_path, sampleRate=44100)()

# Plot a comprehensive diagnostic report
fig, axs = ss.plot_quality_report(audio, 44100, report, title="Antarctic Soundscape Diagnostic")
```

---

## Interactive Jupyter Notebooks

Three Jupyter Notebooks are provided under the `notebooks/` directory:
1. **[01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb)**:
   Loads the baseline files, runs sanity check functions, and displays basic waveform and energy profiles.
2. **[02_sanity_check_demo.ipynb](notebooks/02_sanity_check_demo.ipynb)**:
   Artificially degrades a clean signal with clicks, clipping, silence, and noise, and tests the sanity check functions to verify and visualize detection thresholds.
3. **[03_batch_report.ipynb](notebooks/03_batch_report.ipynb)**:
   Runs the checks over an entire directory, tabulates the results, cross-tabulates folder labels against detected status, and shows how to re-run the corpus with different thresholds.
