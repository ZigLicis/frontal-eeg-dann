#!/usr/bin/env python3
"""
Visualize EEG Time-Series and Frequency Domain Images
====================================================

This script loads test data from your LOSO cross-validation and creates
visualizations showing:
1. Raw time-series EEG (4-second windows)
2. Frequency-domain images (that go into the CNN model)

For both drowsy and normal samples.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports from training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from utils import load_matlab_v73

def reconstruct_time_series(freq_image, fs=250, duration=5):
    """
    Reconstruct approximate time-series from frequency domain image
    Note: This is an approximation since we only have magnitude, not phase
    """
    # freq_image shape: (freq_bins, channels)
    freq_bins, n_channels = freq_image.shape
    
    # Create time vector
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples)
    
    # Reconstruct time series for each channel
    time_series = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Use inverse FFT with random phase (since we only have magnitude)
        # This gives us a plausible time series with correct spectral content
        magnitude = freq_image[:, ch]
        
        # Create full spectrum (mirror for real signal)
        full_spectrum = np.zeros(n_samples, dtype=complex)
        full_spectrum[:freq_bins] = magnitude * np.exp(1j * np.random.uniform(0, 2*np.pi, freq_bins))
        full_spectrum[n_samples-freq_bins+1:] = np.conj(full_spectrum[1:freq_bins][::-1])
        
        # Inverse FFT to get time series
        time_series[ch, :] = np.real(np.fft.ifft(full_spectrum))
    
    return time_series, t

def plot_eeg_comparison(fold_num=3, save_plots=True):
    """
    Create comparison plots of drowsy vs normal EEG samples
    """
    print(f"Loading data from fold {fold_num}...")
    
    # Load fold data
    try:
        fold_data = load_matlab_v73(f'outputs/python_data/fold_{fold_num}_data.mat')
    except Exception as e:
        print(f"Error loading fold {fold_num}: {e}")
        return
    
    # Extract test data
    X_test = fold_data['XTest']  # Shape: (freq, channels, 1, samples)
    y_test = fold_data['YTest_numeric'].flatten()
    
    # Reshape for easier handling: (samples, freq, channels)
    X_test = np.transpose(X_test, (3, 0, 1, 2)).squeeze()  # Remove singleton dimension
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Labels shape: {y_test.shape}")
    print(f"Unique labels: {np.unique(y_test)}")
    
    # Find one sample of each class
    normal_idx = np.where(y_test == 1)[0][0]  # First normal sample
    drowsy_idx = np.where(y_test == 2)[0][0]  # First drowsy sample
    
    print(f"Using normal sample {normal_idx}, drowsy sample {drowsy_idx}")
    
    # Get frequency domain images
    normal_freq = X_test[normal_idx]  # Shape: (freq, channels)
    drowsy_freq = X_test[drowsy_idx]  # Shape: (freq, channels)
    
    # Reconstruct approximate time series
    normal_time, t = reconstruct_time_series(normal_freq)
    drowsy_time, t = reconstruct_time_series(drowsy_freq)
    
    # Channel names (assuming frontal channels)
    channel_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8']
    n_channels = min(len(channel_names), normal_freq.shape[1])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Normal Time Series
    plt.subplot(2, 4, 1)
    for ch in range(n_channels):
        plt.plot(t, normal_time[ch] + ch*50, label=channel_names[ch])
    plt.title('Normal State - Time Series EEG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV) + Offset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Drowsy Time Series  
    plt.subplot(2, 4, 2)
    for ch in range(n_channels):
        plt.plot(t, drowsy_time[ch] + ch*50, label=channel_names[ch])
    plt.title('Drowsy State - Time Series EEG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV) + Offset')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Normal Frequency Domain Image
    plt.subplot(2, 4, 3)
    im1 = plt.imshow(normal_freq.T, aspect='auto', cmap='viridis', origin='lower')
    plt.title('Normal State - Frequency Domain')
    plt.xlabel('Frequency Bins (0-31.25 Hz)')
    plt.ylabel('EEG Channels')
    plt.yticks(range(n_channels), channel_names[:n_channels])
    plt.colorbar(im1, label='Magnitude')
    
    # Plot 4: Drowsy Frequency Domain Image
    plt.subplot(2, 4, 4)
    im2 = plt.imshow(drowsy_freq.T, aspect='auto', cmap='viridis', origin='lower')
    plt.title('Drowsy State - Frequency Domain')
    plt.xlabel('Frequency Bins (0-31.25 Hz)')
    plt.ylabel('EEG Channels')
    plt.yticks(range(n_channels), channel_names[:n_channels])
    plt.colorbar(im2, label='Magnitude')
    
    # Plot 5-8: Individual channel comparisons in frequency domain
    for i, ch_name in enumerate(channel_names[:4]):  # Show first 4 channels
        plt.subplot(2, 4, 5+i)
        # Correct frequency axis calculation
        fs = 250  # Sampling frequency
        fft_length = 1024  # From your preprocessing
        freq_bins = normal_freq.shape[0]  # Should be 128
        max_freq = fs * freq_bins / fft_length  # 250 * 128 / 1024 = 31.25 Hz
        freq_axis = np.linspace(0, max_freq, freq_bins)
        
        plt.plot(freq_axis, normal_freq[:, i], label='Normal', color='blue', alpha=0.7)
        plt.plot(freq_axis, drowsy_freq[:, i], label='Drowsy', color='red', alpha=0.7)
        plt.title(f'{ch_name} - Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'eeg_comparison_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'eeg_comparison_fold_{fold_num}.svg', bbox_inches='tight')
        print(f"Saved plots as eeg_comparison_fold_{fold_num}.png and .svg")
    
    plt.show()
    
    # Print some statistics
    print(f"\n=== Sample Statistics ===")
    print(f"Normal sample - Mean power: {np.mean(normal_freq):.3f}, Max: {np.max(normal_freq):.3f}")
    print(f"Drowsy sample - Mean power: {np.mean(drowsy_freq):.3f}, Max: {np.max(drowsy_freq):.3f}")
    
    # Frequency band analysis
    fs = 250  # Sampling frequency
    fft_length = 1024  # From your preprocessing
    freq_bins = normal_freq.shape[0]  # Should be 128
    max_freq = fs * freq_bins / fft_length  # 250 * 128 / 1024 = 31.25 Hz
    freq_axis = np.linspace(0, max_freq, freq_bins)
    
    delta_band = (freq_axis >= 0.5) & (freq_axis <= 4)
    theta_band = (freq_axis >= 4) & (freq_axis <= 8)
    alpha_band = (freq_axis >= 8) & (freq_axis <= 13)
    beta_band = (freq_axis >= 13) & (freq_axis <= 30)
    
    print(f"\n=== Frequency Band Power (averaged across channels) ===")
    for band_name, band_mask in [('Delta', delta_band), ('Theta', theta_band), 
                                  ('Alpha', alpha_band), ('Beta', beta_band)]:
        normal_power = np.mean(normal_freq[band_mask, :])
        drowsy_power = np.mean(drowsy_freq[band_mask, :])
        print(f"{band_name:>6} - Normal: {normal_power:.3f}, Drowsy: {drowsy_power:.3f}, "
              f"Ratio: {drowsy_power/normal_power:.3f}")

def plot_average_across_folds(data_dir='outputs/python_data', save_plots=True):
    """
    Create comparison plots averaging across ALL folds/subjects
    Layout: Normal Freq Domain | Drowsy Freq Domain | FP2 Spectrum | F7 Spectrum
    """
    import glob
    
    print(f"Loading data from all folds in {data_dir}...")
    
    # Find all fold files
    fold_files = sorted(glob.glob(os.path.join(data_dir, 'fold_*_data.mat')))
    if not fold_files:
        print(f"Error: No fold files found in {data_dir}")
        return
    
    print(f"Found {len(fold_files)} fold files")
    
    # Collect samples from all folds
    all_normal_freq = []
    all_drowsy_freq = []
    
    for fold_file in fold_files:
        try:
            fold_data = load_matlab_v73(fold_file)
            X_test = fold_data['XTest']  # Shape: (freq, channels, 1, samples)
            y_test = fold_data['YTest_numeric'].flatten()
            
            # Reshape: (samples, freq, channels)
            X_test = np.transpose(X_test, (3, 0, 1, 2)).squeeze()
            
            # Collect all normal and drowsy samples
            normal_samples = X_test[y_test == 1]
            drowsy_samples = X_test[y_test == 2]
            
            all_normal_freq.append(normal_samples)
            all_drowsy_freq.append(drowsy_samples)
            
            fold_num = os.path.basename(fold_file).split('_')[1]
            print(f"  Fold {fold_num}: {len(normal_samples)} normal, {len(drowsy_samples)} drowsy samples")
            
        except Exception as e:
            print(f"  Error loading {fold_file}: {e}")
            continue
    
    if not all_normal_freq or not all_drowsy_freq:
        print("Error: No data loaded")
        return
    
    # Concatenate all samples
    all_normal_freq = np.vstack(all_normal_freq)
    all_drowsy_freq = np.vstack(all_drowsy_freq)
    
    print(f"\nTotal samples - Normal: {len(all_normal_freq)}, Drowsy: {len(all_drowsy_freq)}")
    
    # Compute AVERAGE frequency spectrum across all samples
    avg_normal_freq = np.mean(all_normal_freq, axis=0)  # Shape: (freq, channels)
    avg_drowsy_freq = np.mean(all_drowsy_freq, axis=0)
    
    # Also compute std for error bands
    std_normal_freq = np.std(all_normal_freq, axis=0)
    std_drowsy_freq = np.std(all_drowsy_freq, axis=0)
    
    # Channel names and indices
    channel_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8']
    n_channels = min(len(channel_names), avg_normal_freq.shape[1])
    
    # Get indices for FP2 and F7
    fp2_idx = channel_names.index('FP2')  # 1
    f7_idx = channel_names.index('F7')    # 2
    
    # Frequency axis calculation
    fs = 250
    fft_length = 1024
    freq_bins = avg_normal_freq.shape[0]
    max_freq = fs * freq_bins / fft_length
    freq_axis = np.linspace(0, max_freq, freq_bins)
    
    # Create figure with 4 horizontal panels
    # Use gridspec to control relative widths (heatmaps need extra space for colorbars)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), 
                              gridspec_kw={'width_ratios': [1.15, 1.15, 1, 1]})
    
    # Compute shared color scale for both heatmaps
    vmin = min(avg_normal_freq.min(), avg_drowsy_freq.min())
    vmax = max(avg_normal_freq.max(), avg_drowsy_freq.max())
    # Make symmetric around 0 for better visualization
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs
    
    # Panel 1: Alert State - Frequency Domain Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(avg_normal_freq.T, aspect='auto', cmap='viridis', origin='lower',
                     extent=[0, max_freq, -0.5, n_channels-0.5], vmin=vmin, vmax=vmax)
    ax1.set_title('Alert State', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('EEG Channels', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(n_channels))
    ax1.set_yticklabels(channel_names[:n_channels], fontsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.ax.tick_params(labelsize=11)
    
    # Panel 2: Drowsy State - Frequency Domain Heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(avg_drowsy_freq.T, aspect='auto', cmap='viridis', origin='lower',
                     extent=[0, max_freq, -0.5, n_channels-0.5], vmin=vmin, vmax=vmax)
    ax2.set_title('Drowsy State', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('EEG Channels', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(n_channels))
    ax2.set_yticklabels(channel_names[:n_channels], fontsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.ax.tick_params(labelsize=11)
    
    # Panel 3: FP2 Frequency Spectrum
    ax3 = axes[2]
    ax3.fill_between(freq_axis, 
                    avg_normal_freq[:, fp2_idx] - std_normal_freq[:, fp2_idx],
                    avg_normal_freq[:, fp2_idx] + std_normal_freq[:, fp2_idx],
                    color='blue', alpha=0.2)
    ax3.fill_between(freq_axis,
                    avg_drowsy_freq[:, fp2_idx] - std_drowsy_freq[:, fp2_idx],
                    avg_drowsy_freq[:, fp2_idx] + std_drowsy_freq[:, fp2_idx],
                    color='red', alpha=0.2)
    ax3.plot(freq_axis, avg_normal_freq[:, fp2_idx], label='Alert', color='blue', linewidth=2)
    ax3.plot(freq_axis, avg_drowsy_freq[:, fp2_idx], label='Drowsy', color='red', linewidth=2)
    ax3.set_title('FP2 Channel Spectrum', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Magnitude', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=12)
    ax3.legend(fontsize=12, framealpha=0.9, edgecolor='black', fancybox=False)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: F7 Frequency Spectrum
    ax4 = axes[3]
    ax4.fill_between(freq_axis, 
                    avg_normal_freq[:, f7_idx] - std_normal_freq[:, f7_idx],
                    avg_normal_freq[:, f7_idx] + std_normal_freq[:, f7_idx],
                    color='blue', alpha=0.2)
    ax4.fill_between(freq_axis,
                    avg_drowsy_freq[:, f7_idx] - std_drowsy_freq[:, f7_idx],
                    avg_drowsy_freq[:, f7_idx] + std_drowsy_freq[:, f7_idx],
                    color='red', alpha=0.2)
    ax4.plot(freq_axis, avg_normal_freq[:, f7_idx], label='Alert', color='blue', linewidth=2)
    ax4.plot(freq_axis, avg_drowsy_freq[:, f7_idx], label='Drowsy', color='red', linewidth=2)
    ax4.set_title('F7 Channel Spectrum', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Magnitude', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=12)
    ax4.legend(fontsize=12, framealpha=0.9, edgecolor='black', fancybox=False)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(w_pad=2.5)
    
    if save_plots:
        plt.savefig('eeg_spectral_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('eeg_spectral_comparison.svg', bbox_inches='tight')
        print(f"\nSaved plots as eeg_spectral_comparison.png and .svg")
    
    plt.show()
    
    # Print statistics
    print(f"\n=== Average Sample Statistics ===")
    print(f"Normal - Mean power: {np.mean(avg_normal_freq):.3f}, Std: {np.mean(std_normal_freq):.3f}")
    print(f"Drowsy - Mean power: {np.mean(avg_drowsy_freq):.3f}, Std: {np.mean(std_drowsy_freq):.3f}")
    
    # Frequency band analysis
    delta_band = (freq_axis >= 0.5) & (freq_axis <= 4)
    theta_band = (freq_axis >= 4) & (freq_axis <= 8)
    alpha_band = (freq_axis >= 8) & (freq_axis <= 13)
    beta_band = (freq_axis >= 13) & (freq_axis <= 30)
    
    print(f"\n=== Average Frequency Band Power (across all channels & subjects) ===")
    for band_name, band_mask in [('Delta', delta_band), ('Theta', theta_band), 
                                  ('Alpha', alpha_band), ('Beta', beta_band)]:
        normal_power = np.mean(avg_normal_freq[band_mask, :])
        drowsy_power = np.mean(avg_drowsy_freq[band_mask, :])
        diff_pct = 100 * (drowsy_power - normal_power) / normal_power
        print(f"{band_name:>6} - Normal: {normal_power:.3f}, Drowsy: {drowsy_power:.3f}, "
              f"Diff: {diff_pct:+.1f}%")


def main():
    """Main function to create EEG visualizations"""
    print("=== EEG Time-Series and Frequency Domain Visualization ===")
    
    # Check if data exists (outputs/python_data/ relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'outputs', 'python_data')
    if not os.path.exists(data_dir):
        print(f"Error: No exported data found at {data_dir}. Run MATLAB preprocessing first.")
        return
    
    print(f"Using data directory: {data_dir}")
    
    # Generate average plot across all folds
    plot_average_across_folds(data_dir=data_dir, save_plots=True)
    
    print(f"\nVisualization complete! The plots show:")
    print("1. Average time-series EEG for both states (across all subjects)")
    print("2. Average frequency-domain images (input to CNN)")
    print("3. Individual channel frequency spectra with std bands")
    print("4. Frequency band power analysis")

if __name__ == "__main__":
    main() 