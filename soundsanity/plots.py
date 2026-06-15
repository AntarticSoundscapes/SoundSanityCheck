import matplotlib.pyplot as plt
import numpy as np

def plot_waveform_with_issues(audio, sample_rate, click_timestamps=None, saturation_intervals=None, title="Audio Waveform"):
    """
    Plot the audio waveform and overlay detected issues (clicks, clipping).
    
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sampling rate in Hz.
        click_timestamps (list of float, optional): Click timestamps in seconds.
        saturation_intervals (list of tuple, optional): Saturated (start, end) intervals in seconds.
        title (str): Title of the plot.
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axis objects.
    """
    time = np.arange(len(audio)) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot waveform with a premium slate color
    ax.plot(time, audio, color='#2c3e50', alpha=0.75, linewidth=0.8, label="Audio Signal")
    
    # Overlay saturated regions
    if saturation_intervals:
        for idx, (start, end) in enumerate(saturation_intervals):
            label = "Clipping/Saturation" if idx == 0 else ""
            ax.axvspan(start, end, color='#e74c3c', alpha=0.35, label=label)
            
    # Overlay click lines
    if click_timestamps:
        for idx, t in enumerate(click_timestamps):
            label = "Detected Click" if idx == 0 else ""
            ax.axvline(t, color='#e67e22', linestyle='--', linewidth=1.2, alpha=0.8, label=label)
            
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0, time[-1] if len(time) > 0 else 1.0)
    ax.set_ylim(-1.05, 1.05)
    
    # Legend if items are added
    if (click_timestamps and len(click_timestamps) > 0) or (saturation_intervals and len(saturation_intervals) > 0):
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        
    plt.tight_layout()
    return fig, ax


def plot_quality_report(audio, sample_rate, report, title="Audio Sanity Report"):
    """
    Generate a two-panel diagnostic plot summarizing the audio quality report.
    
    Panel 1: Waveform with click/saturation overlays.
    Panel 2: Frame-wise RMS energy (dB) over time, showing the noise floor, 
             active signal level, and silence thresholds.
             
    Args:
        audio (np.ndarray): Audio signal array.
        sample_rate (float): Sampling rate in Hz.
        report (dict): The result dictionary from analyze_recording or individual check dicts.
        title (str): Title of the figure.
        
    Returns:
        tuple: (fig, axs) Matplotlib figure and array of axes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    
    time = np.arange(len(audio)) / sample_rate
    duration = time[-1] if len(time) > 0 else 0.0
    
    # Top Panel: Waveform
    ax1.plot(time, audio, color='#34495e', alpha=0.7, linewidth=0.8, label="Signal")
    
    sat_intervals = report.get("saturation", {}).get("saturation_intervals", [])
    if sat_intervals:
        for idx, (start, end) in enumerate(sat_intervals):
            label = "Saturated Region" if idx == 0 else ""
            ax1.axvspan(start, end, color='#d35400', alpha=0.35, label=label)
            
    click_times = report.get("clicks", {}).get("click_timestamps", [])
    if click_times:
        for idx, t in enumerate(click_times):
            label = "Click" if idx == 0 else ""
            ax1.axvline(t, color='#e74c3c', linestyle='--', linewidth=1.0, alpha=0.8, label=label)
            
    status_colors = {
        "CLEAN": "#2ecc71",
        "SILENT": "#95a5a6",
        "SATURATED": "#d35400",
        "CLICKY": "#e74c3c",
        "NOISY": "#3498db"
    }
    status = report.get("status", "UNKNOWN")
    status_color = status_colors.get(status, "#7f8c8d")
    
    ax1.set_title(f"{title} - File: {report.get('file_name', 'Unknown')} | Status: {status}", 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(loc='upper right')
    
    # Bottom Panel: Energy Profile (dB)
    # Compute frame-wise RMS
    frame_size = 1024
    hop_size = 512
    frame_rms = []
    frame_times = []
    
    # Manual framing to extract frame times
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]
        rms_val = np.sqrt(np.mean(frame**2))
        frame_rms.append(rms_val)
        frame_times.append((i + frame_size/2) / sample_rate)
        
    frame_rms_db = 20 * np.log10(np.array(frame_rms) + 1e-12) if frame_rms else np.array([])
    
    if len(frame_rms_db) > 0:
        ax2.plot(frame_times, frame_rms_db, color='#7f8c8d', alpha=0.8, label="Frame RMS")
        
    # Draw horizontal lines for thresholds and levels
    noise_db = report.get("noise", {}).get("noise_floor_db", -100.0)
    active_db = report.get("noise", {}).get("active_level_db", -100.0)
    
    ax2.axhline(noise_db, color='#2980b9', linestyle=':', linewidth=1.5, 
                label=f"Noise Floor ({noise_db:.1f} dB)")
    ax2.axhline(active_db, color='#27ae60', linestyle='-.', linewidth=1.5, 
                label=f"Active Level ({active_db:.1f} dB)")
    
    # Silence threshold
    # Look up silence threshold from configs or default to -50
    silence_thresh_db = -50.0
    ax2.axhline(silence_thresh_db, color='#95a5a6', linestyle='--', linewidth=1.2, 
                label=f"Silence Threshold ({silence_thresh_db:.1f} dB)")
    
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel("Energy (dB)", fontsize=11)
    ax2.set_ylim(-95, 5)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left', framealpha=0.9)
    
    # Add diagnostic text box in the second plot
    issues = report.get("issues", [])
    issues_text = "\n".join(issues) if issues else "No quality issues detected."
    
    # Color the box based on status
    bbox_props = dict(boxstyle="round,pad=0.5", fc=status_color, alpha=0.15, ec=status_color)
    ax2.text(0.97, 0.9, f"Diagnostics:\n{issues_text}", transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
    
    plt.tight_layout()
    return fig, (ax1, ax2)
