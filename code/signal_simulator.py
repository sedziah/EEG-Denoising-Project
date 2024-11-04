import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set simulation parameters
sampling_rate = 1000  # Sampling rate in Hz
duration = 1  # Duration in seconds
time_points = np.arange(0, duration, 1 / sampling_rate)

# Define parameters for the clean EEG SSVEP signal
A_SIGNAL = 5e-6  # Amplitude of the clean EEG signal in volts (5 microvolts)
F_SIGNAL = 10  # Frequency of the EEG SSVEP signal in Hz (10 Hz)

# Ensure the data folder exists one level up
os.makedirs("../data", exist_ok=True)

def generate_clean_eeg_signal():
    """Generates a clean EEG signal based on SSVEP parameters."""
    return A_SIGNAL * np.sin(2 * np.pi * F_SIGNAL * time_points)

def calculate_power(signal):
    """Calculates the power of a given signal using RMS."""
    return np.mean(signal**2)

def generate_eog_noise(target_snr_db):
    """Generates EOG noise at a target SNR with respect to the clean EEG signal."""
    B_EOG = 1  # Initial amplitude for EOG noise before scaling
    F_EOG = 1  # Frequency of EOG noise in Hz (1 Hz)

    # Initial EOG noise signal
    eog_noise = B_EOG * np.sin(2 * np.pi * F_EOG * time_points)

    # Scale to achieve target SNR
    power_signal = calculate_power(generate_clean_eeg_signal())
    power_noise_target = power_signal / (10 ** (target_snr_db / 10))
    scaling_factor = np.sqrt(power_noise_target / calculate_power(eog_noise))
    return eog_noise * scaling_factor

def generate_emg_noise(target_snr_db):
    """Generates EMG noise at a target SNR with respect to the clean EEG signal."""
    B_EMG = 1  # Initial amplitude for EMG noise before scaling
    F_EMG = 20  # Frequency of EMG noise in Hz (20 Hz)

    # Initial EMG noise signal
    emg_noise = B_EMG * np.sin(2 * np.pi * F_EMG * time_points)

    # Scale to achieve target SNR
    power_signal = calculate_power(generate_clean_eeg_signal())
    power_noise_target = power_signal / (10 ** (target_snr_db / 10))
    scaling_factor = np.sqrt(power_noise_target / calculate_power(emg_noise))
    return emg_noise * scaling_factor

def save_signals_to_csv(clean_signal, eog_noise, emg_noise):
    """Saves the clean EEG signal, EOG noise, EMG noise, and composite signal to a CSV file."""
    composite_signal = clean_signal + eog_noise + emg_noise
    data = {
        "Time (s)": time_points,
        "Clean EEG Signal (µV)": clean_signal * 1e6,  # Convert to microvolts
        "EOG Noise (µV)": eog_noise * 1e6,
        "EMG Noise (µV)": emg_noise * 1e6,
        "Composite Signal (µV)": composite_signal * 1e6
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join("../data", "simulated_eeg_data.csv")  # Save in ../data
    df.to_csv(csv_path, index=False)
    print(f"Signals saved to {csv_path}")

def plot_and_save_signals(clean_signal, eog_noise, emg_noise):
    """Plots and saves the clean EEG signal, EOG noise, EMG noise, and the composite signal as images."""
    composite_signal = clean_signal + eog_noise + emg_noise

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(time_points, clean_signal * 1e6, label="Clean EEG Signal (10 Hz)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.savefig("../data/clean_eeg_signal.png")  # Save in ../data

    plt.subplot(4, 1, 2)
    plt.plot(time_points, eog_noise * 1e6, label="EOG Noise (1 Hz)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.savefig("../data/eog_noise.png")  # Save in ../data

    plt.subplot(4, 1, 3)
    plt.plot(time_points, emg_noise * 1e6, label="EMG Noise (20 Hz)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.savefig("../data/emg_noise.png")  # Save in ../data

    plt.subplot(4, 1, 4)
    plt.plot(time_points, composite_signal * 1e6, label="Composite Signal (EEG + EOG + EMG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.savefig("../data/composite_signal.png")  # Save in ../data

    plt.tight_layout()
    plt.show()
    print("Plots saved as images in the '../data' folder.")

# Example usage
if __name__ == "__main__":
    clean_signal = generate_clean_eeg_signal()
    eog_noise = generate_eog_noise(-3)  # Example SNR for EOG noise
    emg_noise = generate_emg_noise(-5)  # Example SNR for EMG noise

    # Save signals to CSV
    save_signals_to_csv(clean_signal, eog_noise, emg_noise)

    # Plot and save signals as images
    plot_and_save_signals(clean_signal, eog_noise, emg_noise)
