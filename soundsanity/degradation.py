import numpy as np

def add_clicks(audio, num_clicks=50, click_amplitude=0.8, sample_rate=44100):
    """
    Artificially inject impulsive clicks into an audio signal.
    
    Clicks are modeled as short biphasic impulse spikes.
    
    Args:
        audio (np.ndarray): Original clean audio signal.
        num_clicks (int): Number of clicks to inject.
        click_amplitude (float): Peak amplitude of the click impulse.
        sample_rate (float): Sample rate in Hz.
        
    Returns:
        tuple: (degraded_audio, click_timestamps)
            - degraded_audio (np.ndarray): Audio with injected clicks.
            - click_timestamps (list of float): Absolute timestamps (in seconds) of injected clicks.
    """
    degraded = audio.copy()
    n_samples = len(audio)
    
    if n_samples < 50:
        return degraded, []
        
    # Choose random, non-contiguous sample indices for clicks
    click_indices = sorted(np.random.choice(np.arange(10, n_samples - 10), size=num_clicks, replace=False))
    click_timestamps = [float(idx) / sample_rate for idx in click_indices]
    
    for idx in click_indices:
        # Biphasic impulse
        degraded[idx] = click_amplitude
        degraded[idx+1] = -click_amplitude * 0.8
        degraded[idx+2] = click_amplitude * 0.4
        
    return degraded, click_timestamps


def add_clipping(audio, clip_threshold=0.01):
    """
    Artificially saturate (clip) the audio signal by flat-lining samples exceeding a threshold.
    
    Args:
        audio (np.ndarray): Original clean audio signal.
        clip_threshold (float): Amplitude above which to clip.
        
    Returns:
        np.ndarray: Clipped and boosted audio signal.
    """
    # Normalize audio first to peak at 1.0 to make clipping threshold predictable
    peak = np.max(np.abs(audio))
    if peak > 0:
        normalized = audio / peak
    else:
        normalized = audio.copy()
        
    # Clip the signal
    clipped = np.clip(normalized, -clip_threshold, clip_threshold)
    
    # Scale back up to make it audibly active (boosted to peak at 1.0)
    if clip_threshold > 0:
        clipped = clipped / clip_threshold
        
    return clipped


def add_noise(audio, target_snr_db=10):
    """
    Add white Gaussian noise to the audio signal to achieve a target Signal-to-Noise Ratio (SNR).
    
    Args:
        audio (np.ndarray): Original clean audio signal.
        target_snr_db (float): Target SNR in decibels.
        
    Returns:
        np.ndarray: Noisy audio signal.
    """
    sig_power = np.mean(audio ** 2)
    if sig_power == 0:
        sig_power = 1e-12
        
    # Generate white Gaussian noise
    noise = np.random.randn(*audio.shape).astype(np.float32)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        noise_power = 1e-12
        
    # Calculate noise scaling factor
    # SNR = SigPower / NoisePower
    # NoisePower_target = SigPower / (10^(SNR_dB/10))
    target_noise_power = sig_power / (10 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    
    noisy_audio = audio + noise * scale
    
    # Normalize if it exceeds 1.0 to prevent hardware clipping
    peak = np.max(np.abs(noisy_audio))
    if peak > 1.0:
        noisy_audio = noisy_audio / peak
        
    return noisy_audio


def make_silent(audio, level_db=-90):
    """
    Artificially quiet/silence the audio signal to mock a recording hardware failure.
    
    Args:
        audio (np.ndarray): Original clean audio signal.
        level_db (float): Target RMS level in decibels.
        
    Returns:
        np.ndarray: Attenuated audio signal.
    """
    scale = 10 ** (level_db / 20.0)
    return audio * scale
