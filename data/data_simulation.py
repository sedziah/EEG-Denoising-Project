import numpy as np
import matplotlib.pyplot as plt

# Set simulation parameters
sampling_rate = 1000  # 1000 Hz
duration = 1  # 1 second
time_points = np.arange(0, duration, 1 / sampling_rate)

# Parameters for the Clean EEG SSVEP Signal (10 Hz, 5 microvolts)
A_signal = 5e-6  # Amplitude of the clean EEG signal (5 microvolts)
f_signal = 10  # Frequency of the EEG SSVEP signal (10 Hz)

# Generate the Clean EEG Signal
clean_eeg_signal = A_signal * np.sin(2 * np.pi * f_signal * time_points)

# Helper function to calculate RMS-based power
def calculate_power(signal):
    return np.mean(signal**2)

# Generate EOG Noise (1 Hz, low-frequency, adjustable to SNR of -7 dB to 2 dB)
def generate_eog_noise(target_snr_db):
    B_eog = 1  # Initial amplitude for the EOG noise (to be scaled)
    f_eog = 1  # Frequency of EOG noise (1 Hz)

    # Initial EOG noise before scaling
    eog_noise = B_eog * np.sin(2 * np.pi * f_eog * time_points)

    # Calculate required noise power
    power_signal = calculate_power(clean_eeg_signal)
    power_noise_target = power_signal / (10 ** (target_snr_db / 10))

    # Scale EOG noise to match target power
    scaling_factor = np.sqrt(power_noise_target / calculate_power(eog_noise))
    scaled_eog_noise = eog_noise * scaling_factor
    return scaled_eog_noise

# Generate EMG Noise (20 Hz, higher-frequency, adjustable to SNR of -7 dB to 4 dB)
def generate_emg_noise(target_snr_db):
    B_emg = 1  # Initial amplitude for the EMG noise (to be scaled)
    f_emg = 20  # Frequency of EMG noise (20 Hz)

    # Initial EMG noise before scaling
    emg_noise = B_emg * np.sin(2 * np.pi * f_emg * time_points)

    # Calculate required noise power
    power_signal = calculate_power(clean_eeg_signal)
    power_noise_target = power_signal / (10 ** (target_snr_db / 10))

    # Scale EMG noise to match target power
    scaling_factor = np.sqrt(power_noise_target / calculate_power(emg_noise))
    scaled_emg_noise = emg_noise * scaling_factor
    return scaled_emg_noise

# Generate and plot signals for EOG and EMG at specific SNRs
eog_noise = generate_eog_noise(-3)  # Example SNR for EOG noise
emg_noise = generate_emg_noise(-5)  # Example SNR for EMG noise

# Composite Signal: Clean EEG + EOG Noise + EMG Noise
composite_signal = clean_eeg_signal + eog_noise + emg_noise

# Plot the signals
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(time_points, clean_eeg_signal * 1e6, label='Clean EEG Signal (10 Hz)')
plt.ylabel("Amplitude (µV)")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_points, eog_noise * 1e6, label='EOG Noise (1 Hz)')
plt.ylabel("Amplitude (µV)")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_points, emg_noise * 1e6, label='EMG Noise (20 Hz)')
plt.ylabel("Amplitude (µV)")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_points, composite_signal * 1e6, label='Composite Signal (EEG + EOG + EMG)')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()

plt.tight_layout()
plt.show()
