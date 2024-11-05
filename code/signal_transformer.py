#signal_transformer.py

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import os

# Ensure the data folder exists one level up
os.makedirs("../data", exist_ok=True)

def compute_frequency_domain(signal, sampling_rate):
    """
    Computes the FFT of a signal and returns the frequency bins and amplitudes.

    Args:
        signal (np.ndarray): The time-domain signal to be transformed.
        sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
        tuple: Frequencies and amplitudes for saving to CSV.
    """
    N = len(signal)  # Number of samples
    T = 1 / sampling_rate  # Sampling interval
    yf = fft(signal)  # Compute the FFT
    xf = fftfreq(N, T)[:N // 2]  # Frequency bins
    amplitude = 2.0 / N * np.abs(yf[0:N // 2])  # Amplitude

    return xf, amplitude

def save_combined_frequency_data(clean_signal, eog_noise, emg_noise, sampling_rate):
    """
    Saves the frequency domain data of clean EEG, EOG noise, and EMG noise into a single CSV file.

    Args:
        clean_signal (np.ndarray): Clean EEG signal.
        eog_noise (np.ndarray): EOG noise signal.
        emg_noise (np.ndarray): EMG noise signal.
        sampling_rate (int): The sampling rate of the signals in Hz.
    """
    # Get frequency and amplitude for each signal
    xf_clean, amp_clean = compute_frequency_domain(clean_signal, sampling_rate)
    xf_eog, amp_eog = compute_frequency_domain(eog_noise, sampling_rate)
    xf_emg, amp_emg = compute_frequency_domain(emg_noise, sampling_rate)

    # Create a combined DataFrame
    combined_data = {
        "Frequency (Hz)": xf_clean,  # Assuming all signals have the same frequency bins
        "Clean EEG Amplitude": amp_clean,
        "EOG Amplitude": np.interp(xf_clean, xf_eog, amp_eog, left=0, right=0),  # Interpolating EOG onto clean frequencies
        "EMG Amplitude": np.interp(xf_clean, xf_emg, amp_emg, left=0, right=0)   # Interpolating EMG onto clean frequencies
    }
    df_combined = pd.DataFrame(combined_data)
    csv_path = os.path.join("../data", "combined_frequency_domain_data.csv")
    df_combined.to_csv(csv_path, index=False)
    print(f"Combined frequency domain data saved to {csv_path}")

if __name__ == "__main__":
    # Example usage
    from signal_simulator import generate_clean_eeg_signal, generate_eog_noise, generate_emg_noise

    # Generate signals
    clean_signal = generate_clean_eeg_signal()
    eog_noise = generate_eog_noise(-3)  # Example SNR for EOG noise
    emg_noise = generate_emg_noise(-5)  # Example SNR for EMG noise

    # Save combined frequency domain data
    save_combined_frequency_data(clean_signal, eog_noise, emg_noise, 1000)  # Sampling rate is 1000 Hz
