#signal_transformer.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
import os

# Ensure the data folder exists one level up
os.makedirs("../data", exist_ok=True)

def plot_frequency_domain(signal, sampling_rate, signal_name):
    """
    Computes the FFT of a signal, plots the frequency spectrum,
    and saves the frequency data to a CSV file.
    
    Args:
        signal (np.ndarray): The time-domain signal to be transformed.
        sampling_rate (int): The sampling rate of the signal in Hz.
        signal_name (str): The name of the signal for labeling and saving.
    """
    N = len(signal)  # Number of samples
    T = 1 / sampling_rate  # Sampling interval
    yf = fft(signal)  # Compute the FFT
    xf = fftfreq(N, T)[:N // 2]  # Frequency bins

    # Plotting the frequency spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), label='Frequency Spectrum')
    plt.title(f'Frequency Domain Representation of {signal_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 50)  # Limit x-axis for better visibility
    plt.legend()
    plt.grid()
    plt.show()

    # Save the frequency domain data to CSV
    frequency_data = {
        "Frequency (Hz)": xf,
        "Amplitude": 2.0 / N * np.abs(yf[0:N // 2])
    }
    df = pd.DataFrame(frequency_data)
    csv_path = os.path.join("../data", f"{signal_name}_frequency_domain.csv")
    df.to_csv(csv_path, index=False)
    print(f"Frequency domain data saved to {csv_path}")

if __name__ == "__main__":
    # Example usage
    from signal_simulator import generate_clean_eeg_signal, generate_eog_noise, generate_emg_noise

    # Generate signals
    clean_signal = generate_clean_eeg_signal()
    eog_noise = generate_eog_noise(-3)  # Example SNR for EOG noise
    emg_noise = generate_emg_noise(-5)  # Example SNR for EMG noise

    # Plot frequency domain representation and save data for each signal
    plot_frequency_domain(clean_signal, 1000, "Clean_EEG_Signal")  # Sampling rate is 1000 Hz
    plot_frequency_domain(eog_noise, 1000, "EOG_Noise")
    plot_frequency_domain(emg_noise, 1000, "EMG_Noise")
