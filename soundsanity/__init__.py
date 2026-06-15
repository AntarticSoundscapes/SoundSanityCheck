"""
soundsanity: Quality checks and sanity checks for Antarctic soundscape field recordings using Essentia.
"""

from .analysis import (
    detect_clicks,
    detect_saturation,
    detect_silence,
    estimate_noise,
    analyze_recording,
    DEFAULT_CONFIG
)

from .degradation import (
    add_clicks,
    add_clipping,
    add_noise,
    make_silent
)

from .plots import (
    plot_waveform_with_issues,
    plot_quality_report
)

__all__ = [
    'detect_clicks',
    'detect_saturation',
    'detect_silence',
    'estimate_noise',
    'analyze_recording',
    'DEFAULT_CONFIG',
    'add_clicks',
    'add_clipping',
    'add_noise',
    'make_silent',
    'plot_waveform_with_issues',
    'plot_quality_report'
]
