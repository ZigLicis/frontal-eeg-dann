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

def plot_eeg_comparison(fold_num=2, save_plots=True):
    """
    Create comparison plots of drowsy vs normal EEG samples
    """
    print(f"Loading data from fold {fold_num}...")
    
    # Load fold data
    try:
        fold_data = load_matlab_v73(f'diagnostics/python_data/fold_{fold_num}_data.mat')
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
    
    # Find all samples of each class
    normal_indices = np.where(y_test == 1)[0]
    drowsy_indices = np.where(y_test == 2)[0]
    
    # Number of samples to average (use all available or cap at n_samples_to_avg)
    n_samples_to_avg = 50  # Average up to 50 samples per class
    n_normal = min(len(normal_indices), n_samples_to_avg)
    n_drowsy = min(len(drowsy_indices), n_samples_to_avg)
    
    print(f"Averaging {n_normal} normal samples and {n_drowsy} drowsy samples")
    
    # Average frequency domain images across multiple samples
    normal_freq = np.mean(X_test[normal_indices[:n_normal]], axis=0)  # Shape: (freq, channels)
    drowsy_freq = np.mean(X_test[drowsy_indices[:n_drowsy]], axis=0)  # Shape: (freq, channels)
    
    # Reconstruct approximate time series (from averaged spectrum)
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
    
    # Plot 5-8: Individual channel comparisons in frequency domain (overlapped)
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
    
    # === Create separate Normal and Drowsy plots (same style as bottom row) ===
    # Calculate frequency axis once
    fs = 250
    fft_length = 1024
    freq_bins = normal_freq.shape[0]
    max_freq = fs * freq_bins / fft_length
    freq_axis = np.linspace(0, max_freq, freq_bins)
    
    # Get y-axis limits for consistent scaling across both plots
    y_max = max(np.max(normal_freq[:, :4]), np.max(drowsy_freq[:, :4])) * 1.1
    y_min = min(np.min(normal_freq[:, :4]), np.min(drowsy_freq[:, :4])) * 1.1
    
    # === NORMAL STATE PLOT ===
    fig_normal, axes_normal = plt.subplots(1, 4, figsize=(16, 4))
    fig_normal.suptitle('Normal State - Frequency Spectrum', fontsize=14, fontweight='bold')
    
    for i, ch_name in enumerate(channel_names[:4]):
        axes_normal[i].plot(freq_axis, normal_freq[:, i], color='blue', alpha=0.7)
        axes_normal[i].set_title(f'{ch_name} - Frequency Spectrum')
        axes_normal[i].set_xlabel('Frequency (Hz)')
        axes_normal[i].set_ylabel('Magnitude')
        axes_normal[i].set_ylim([y_min, y_max])
        axes_normal[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'eeg_normal_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'eeg_normal_fold_{fold_num}.svg', bbox_inches='tight')
        print(f"Saved: eeg_normal_fold_{fold_num}.png")
    
    plt.show()
    
    # === DROWSY STATE PLOT ===
    fig_drowsy, axes_drowsy = plt.subplots(1, 4, figsize=(16, 4))
    fig_drowsy.suptitle('Drowsy State - Frequency Spectrum', fontsize=14, fontweight='bold')
    
    for i, ch_name in enumerate(channel_names[:4]):
        axes_drowsy[i].plot(freq_axis, drowsy_freq[:, i], color='red', alpha=0.7)
        axes_drowsy[i].set_title(f'{ch_name} - Frequency Spectrum')
        axes_drowsy[i].set_xlabel('Frequency (Hz)')
        axes_drowsy[i].set_ylabel('Magnitude')
        axes_drowsy[i].set_ylim([y_min, y_max])
        axes_drowsy[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'eeg_drowsy_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'eeg_drowsy_fold_{fold_num}.svg', bbox_inches='tight')
        print(f"Saved: eeg_drowsy_fold_{fold_num}.png")
    
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

def main():
    """Main function to create EEG visualizations"""
    print("=== EEG Time-Series and Frequency Domain Visualization ===")
    
    # Check if data exists
    if not os.path.exists('diagnostics/python_data'):
        print("Error: No exported data found. Run MATLAB preprocessing first.")
        return
    
    # You can change the fold number to visualize different subjects
    fold_num = 1  # Change this to visualize different subjects
    
    plot_eeg_comparison(fold_num=fold_num, save_plots=True)
    
    print(f"\nVisualization complete! The plots show:")
    print("1. Raw time-series EEG for both states")
    print("2. Frequency-domain images (input to CNN)")
    print("3. Individual channel frequency spectra")
    print("4. Frequency band power analysis")

if __name__ == "__main__":
    main() 