"""
soundsanity: Quality checks and sanity checks for Antarctic soundscape field recordings using Essentia.
"""

from .analysis import (
    analyze_audio,
    detect_clicks,
    detect_saturation,
    detect_silence,
    estimate_noise,
    analyze_recording,
    DEFAULT_CONFIG
)

from .batch import (
    find_audio_files,
    analyze_directory,
    flatten_report,
    summarize,
    DEFAULT_AUDIO_EXTENSIONS
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

from .corpus import (
    describe_file,
    inventory,
    parse_season,
    parse_site,
    parse_timestamp,
    GAIN_OFFSET_DB
)

from .wind import (
    analyze_wind,
    WIND_CONFIG
)

from .field import (
    analyze_field_recording,
    FIELD_FIELDS
)

from .meteo import (
    load_meteo,
    join_meteo,
    verify_time_alignment,
    WIND_BANDS,
    WIND_BAND_LABELS,
    WIND_SCALE_BREAK
)

from .report import (
    apply_gain_correction,
    add_wind_percentile_bands,
    wind_response,
    standardize_by_wind,
    deployment_quality,
    direction_response
)

__all__ = [
    'analyze_audio',
    'detect_clicks',
    'detect_saturation',
    'detect_silence',
    'estimate_noise',
    'analyze_recording',
    'DEFAULT_CONFIG',
    'find_audio_files',
    'analyze_directory',
    'flatten_report',
    'summarize',
    'DEFAULT_AUDIO_EXTENSIONS',
    'add_clicks',
    'add_clipping',
    'add_noise',
    'make_silent',
    'plot_waveform_with_issues',
    'plot_quality_report',
    'describe_file',
    'inventory',
    'parse_season',
    'parse_site',
    'parse_timestamp',
    'GAIN_OFFSET_DB',
    'analyze_wind',
    'WIND_CONFIG',
    'analyze_field_recording',
    'FIELD_FIELDS',
    'load_meteo',
    'join_meteo',
    'verify_time_alignment',
    'WIND_BANDS',
    'WIND_BAND_LABELS',
    'WIND_SCALE_BREAK',
    'apply_gain_correction',
    'add_wind_percentile_bands',
    'wind_response',
    'standardize_by_wind',
    'deployment_quality',
    'direction_response'
]
