import os
import numpy as np
import essentia.standard as es

# Default configuration parameters for quality checks
DEFAULT_CONFIG = {
    # Click detection config
    "click_detection_threshold": 30.0,
    "max_clicks_rate": 1.0,               # Clicks/sec above which audio is flagged as 'clicky'
    "click_merge_tolerance": 0.01,        # Seconds within which clicks are grouped as one event
    
    # Saturation/Clipping config
    "saturation_energy_threshold": -1.0,  # dB threshold for sample energy in saturated region
    "saturation_diff_threshold": 0.001,   # Minimum difference between consecutive saturated samples
    "saturation_min_duration": 1.0,       # Minimum duration of saturated region in ms
                                          # (5.0 ms misses peak clipping: clipped runs on
                                          #  wind-buffeted recordings top out around 2.5 ms)
    "max_saturation_ratio": 0.001,        # Ratio of clipping above which audio is flagged as 'saturated' (0.1%)
    
    # Silence/Failed recording config
    "silence_threshold_db": -50.0,        # dB level below which a frame is considered silent
    "min_silence_ratio": 0.95,            # Ratio of silence above which audio is flagged as 'silent' (recording failure)
    
    # Noise/SNR estimation config
    "noise_floor_percentile": 10.0,       # Percentile of frame RMS values used as noise floor estimate
    "active_signal_percentile": 90.0,     # Percentile of frame RMS values used as active signal estimate
    "max_noise_floor_db": -45.0,          # Noise floor above this is flagged as 'noisy'
    "min_snr_db": 10.0,                   # SNR below this is flagged as 'noisy'
    
    # Framing config (standard for 44.1kHz audio)
    "frame_size": 1024,
    "hop_size": 512,

    # Rate the loader resamples to. Set this to the recording's native rate to
    # skip resampling entirely (the Antarctic field corpus is 48 kHz).
    "sample_rate": 44100,
}


def merge_intervals(intervals, gap_tolerance=0.0):
    """
    Merge overlapping intervals.
    
    Args:
        intervals (list of tuple): List of (start, end) timestamps.
        gap_tolerance (float): Maximum gap in seconds to merge adjacent intervals.
        
    Returns:
        list of tuple: Merged (start, end) intervals.
    """
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    
    for current in sorted_intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1] + gap_tolerance:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(list(current))
            
    return [tuple(x) for x in merged]


def merge_clicks(click_times, tolerance=0.01):
    """
    Deduplicate and group closely spaced click detections into single click events.
    
    Args:
        click_times (list): Click timestamps in seconds.
        tolerance (float): Grouping tolerance in seconds.
        
    Returns:
        list: Grouped/merged click timestamps.
    """
    if not click_times:
        return []
    
    sorted_times = sorted(click_times)
    merged = []
    current_group = [sorted_times[0]]
    
    for t in sorted_times[1:]:
        if t - current_group[-1] <= tolerance:
            current_group.append(t)
        else:
            merged.append(float(np.mean(current_group)))
            current_group = [t]
            
    if current_group:
        merged.append(float(np.mean(current_group)))
        
    return merged


def detect_clicks(audio, sample_rate, config=None):
    """
    Detect clicks in an audio signal using Essentia's ClickDetector.
    
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Custom configuration override.
        
    Returns:
        dict: A dictionary containing click detection metrics:
            - clicks_count (int)
            - clicks_rate (float, clicks/sec)
            - click_timestamps (list of float)
            - is_clicky (bool)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    duration = len(audio) / sample_rate
    
    if duration == 0:
        return {
            "clicks_count": 0,
            "clicks_rate": 0.0,
            "click_timestamps": [],
            "is_clicky": False
        }
        
    click_detector = es.ClickDetector(
        detectionThreshold=cfg["click_detection_threshold"],
        frameSize=cfg["frame_size"],
        hopSize=cfg["hop_size"],
        sampleRate=sample_rate
    )
    
    raw_clicks = []
    for frame in es.FrameGenerator(audio, frameSize=cfg["frame_size"], hopSize=cfg["hop_size"]):
        starts, _ = click_detector(frame)
        if len(starts) > 0:
            raw_clicks.extend(starts)
            
    # Group and deduplicate clicks since overlapping frames will detect the same click
    merged_clicks = merge_clicks(raw_clicks, tolerance=cfg["click_merge_tolerance"])
    clicks_count = len(merged_clicks)
    clicks_rate = clicks_count / duration
    is_clicky = clicks_rate > cfg["max_clicks_rate"]
    
    return {
        "clicks_count": clicks_count,
        "clicks_rate": clicks_rate,
        "click_timestamps": merged_clicks,
        "is_clicky": bool(is_clicky)
    }


def detect_saturation(audio, sample_rate, config=None):
    """
    Detect saturated/clipped regions in an audio signal using Essentia's SaturationDetector.
    
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Custom configuration override.
        
    Returns:
        dict: A dictionary containing saturation detection metrics:
            - saturation_intervals (list of tuple)
            - saturation_duration (float, total seconds saturated)
            - saturation_ratio (float, proportion of saturated audio)
            - is_saturated (bool)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    duration = len(audio) / sample_rate
    
    if duration == 0:
        return {
            "saturation_intervals": [],
            "saturation_duration": 0.0,
            "saturation_ratio": 0.0,
            "is_saturated": False
        }
        
    sat_detector = es.SaturationDetector(
        energyThreshold=cfg["saturation_energy_threshold"],
        differentialThreshold=cfg["saturation_diff_threshold"],
        minimumDuration=cfg["saturation_min_duration"],
        frameSize=cfg["frame_size"],
        hopSize=cfg["hop_size"],
        sampleRate=sample_rate
    )
    
    raw_intervals = []
    for frame in es.FrameGenerator(audio, frameSize=cfg["frame_size"], hopSize=cfg["hop_size"]):
        starts, ends = sat_detector(frame)
        if len(starts) > 0:
            for s, e in zip(starts, ends):
                raw_intervals.append((s, e))
                
    # Merge overlapping intervals across frames
    merged_intervals = merge_intervals(raw_intervals)
    
    # Calculate duration of saturation
    saturation_duration = sum([e - s for s, e in merged_intervals])
    saturation_ratio = saturation_duration / duration
    is_saturated = saturation_ratio > cfg["max_saturation_ratio"]
    
    return {
        "saturation_intervals": merged_intervals,
        "saturation_duration": float(saturation_duration),
        "saturation_ratio": float(saturation_ratio),
        "is_saturated": bool(is_saturated)
    }


def detect_silence(audio, sample_rate, config=None):
    """
    Detect if the audio recording is silent/empty (indicative of recording failure).
    
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Custom configuration override.
        
    Returns:
        dict: A dictionary containing silence detection metrics:
            - silence_ratio (float, ratio of silent frames to total frames)
            - is_silent (bool, True if silence_ratio exceeds the threshold)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # Convert dB threshold to linear power: Power = 10^(dB/10)
    lin_threshold = 10 ** (cfg["silence_threshold_db"] / 10.0)
    
    silence_rate_detector = es.SilenceRate(thresholds=[lin_threshold])
    
    silent_frames_count = 0
    total_frames = 0
    
    for frame in es.FrameGenerator(audio, frameSize=cfg["frame_size"], hopSize=cfg["hop_size"]):
        outputs = silence_rate_detector(frame)
        if isinstance(outputs, (list, tuple, np.ndarray)):
            silent_frames_count += int(outputs[0])
        else:
            silent_frames_count += int(outputs)
        total_frames += 1
        
    if total_frames == 0:
        return {
            "silence_ratio": 1.0,
            "is_silent": True
        }
        
    silence_ratio = silent_frames_count / total_frames
    is_silent = silence_ratio > cfg["min_silence_ratio"]
    
    return {
        "silence_ratio": float(silence_ratio),
        "is_silent": bool(is_silent)
    }


def estimate_noise(audio, sample_rate, config=None):
    """
    Estimate the background noise floor and Signal-to-Noise Ratio (SNR) indices.
    
    This is based on the distribution of frame-wise Root Mean Square (RMS) energy.
    The noise floor is estimated using a lower percentile (e.g. 10th percentile)
    of the frame energies, which is robust against transient sounds.
    
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Custom configuration override.
        
    Returns:
        dict: A dictionary containing noise/SNR metrics:
            - rms_db (float, overall RMS level in decibels)
            - noise_floor_db (float, estimated noise floor in decibels)
            - active_level_db (float, estimated active signal level in decibels)
            - snr_db (float, active signal minus noise floor)
            - is_noisy (bool, True if noise floor is too high or SNR is too low)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    if len(audio) == 0:
        return {
            "rms_db": -100.0,
            "noise_floor_db": -100.0,
            "active_level_db": -100.0,
            "snr_db": 0.0,
            "is_noisy": True
        }
        
    # Overall RMS in dB
    overall_rms = float(es.RMS()(audio))
    rms_db = 20 * np.log10(overall_rms + 1e-12)
    
    # Frame-wise RMS computation
    frame_rms_values = []
    rms_alg = es.RMS()
    for frame in es.FrameGenerator(audio, frameSize=cfg["frame_size"], hopSize=cfg["hop_size"]):
        frame_rms_values.append(float(rms_alg(frame)))
        
    if not frame_rms_values:
        return {
            "rms_db": rms_db,
            "noise_floor_db": -100.0,
            "active_level_db": -100.0,
            "snr_db": 0.0,
            "is_noisy": True
        }
        
    frame_rms_db = 20 * np.log10(np.array(frame_rms_values) + 1e-12)
    
    # Robust noise floor and active signal levels using percentiles
    noise_floor_db = float(np.percentile(frame_rms_db, cfg["noise_floor_percentile"]))
    active_level_db = float(np.percentile(frame_rms_db, cfg["active_signal_percentile"]))
    
    # Estimated SNR
    snr_db = active_level_db - noise_floor_db
    
    # Check flags
    is_noisy = (noise_floor_db > cfg["max_noise_floor_db"]) or (snr_db < cfg["min_snr_db"])
    
    return {
        "rms_db": float(rms_db),
        "noise_floor_db": float(noise_floor_db),
        "active_level_db": float(active_level_db),
        "snr_db": float(snr_db),
        "is_noisy": bool(is_noisy)
    }


def analyze_recording(file_path, config=None):
    """
    Perform a complete quality analysis on an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        config (dict, optional): Custom configuration overrides.
        
    Returns:
        dict: A dictionary containing the analysis results and status:
            - file_name (str)
            - duration (float)
            - clicks (dict, outputs of detect_clicks)
            - saturation (dict, outputs of detect_saturation)
            - silence (dict, outputs of detect_silence)
            - noise (dict, outputs of estimate_noise)
            - status (str, overall classification: 'CLEAN', 'SILENT', 'SATURATED', 'CLICKY', 'NOISY')
            - issues (list of str, human-readable descriptions of failed checks)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    # Load audio (downmixed to mono, resampled to cfg["sample_rate"])
    sample_rate = cfg["sample_rate"]
    loader = es.MonoLoader(filename=file_path, sampleRate=sample_rate)
    audio = loader()
    
    report = analyze_audio(audio, sample_rate, cfg)
    report["file_name"] = os.path.basename(file_path)
    return report


def analyze_audio(audio, sample_rate, config=None):
    """
    Perform the full quality analysis on an already-loaded signal.

    This is the body of `analyze_recording` without the file loading, for
    callers that hold the samples already and would otherwise decode the file a
    second time.

    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sample rate of the audio in Hz.
        config (dict, optional): Custom configuration overrides.

    Returns:
        dict: The same report `analyze_recording` returns, minus ``file_name``.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    duration = len(audio) / sample_rate
    
    # Run individual diagnostics
    click_res = detect_clicks(audio, sample_rate, cfg)
    sat_res = detect_saturation(audio, sample_rate, cfg)
    silence_res = detect_silence(audio, sample_rate, cfg)
    noise_res = estimate_noise(audio, sample_rate, cfg)
    
    # Determine issues and status
    issues = []
    if silence_res["is_silent"]:
        issues.append(f"Recording is silent (silence ratio {silence_res['silence_ratio']:.2%})")
    if sat_res["is_saturated"]:
        issues.append(f"Signal is saturated (saturated duration {sat_res['saturation_duration']:.3f}s)")
    if click_res["is_clicky"]:
        issues.append(f"Too many clicks detected ({click_res['clicks_count']} clicks, {click_res['clicks_rate']:.2f} clicks/sec)")
    if noise_res["is_noisy"]:
        issues.append(f"Recording is noisy (noise floor {noise_res['noise_floor_db']:.1f} dB, SNR {noise_res['snr_db']:.1f} dB)")
        
    if not issues:
        status = "CLEAN"
    elif silence_res["is_silent"]:
        status = "SILENT"
    elif sat_res["is_saturated"]:
        status = "SATURATED"
    elif click_res["is_clicky"]:
        status = "CLICKY"
    else:
        status = "NOISY"
        
    return {
        "duration": float(duration),
        "clicks": click_res,
        "saturation": sat_res,
        "silence": silence_res,
        "noise": noise_res,
        "status": status,
        "issues": issues
    }
