# main.py

from signal_simulator import (
    generate_clean_eeg_signal,
    generate_eog_noise,
    generate_emg_noise,
    save_signals_to_csv,
    plot_and_save_signals,
)
from signal_transformer import plot_frequency_domain  # Importing the function for frequency analysis

# Generate signals
clean_signal = generate_clean_eeg_signal()
eog_noise = generate_eog_noise(-3)  # Adjust SNR as needed
emg_noise = generate_emg_noise(-5)  # Adjust SNR as needed

# Save signals to CSV
try:
    save_signals_to_csv(clean_signal, eog_noise, emg_noise)
except Exception as e:
    print(f"Error saving CSV: {e}")

# Plot and save signals as images
plot_and_save_signals(clean_signal, eog_noise, emg_noise)

# Plot frequency domain representations
print("Transforming signals to frequency domain...")
plot_frequency_domain(clean_signal, 1000, "Clean_EEG_Signal")  # Sampling rate is 1000 Hz
plot_frequency_domain(eog_noise, 1000, "EOG_Noise")
plot_frequency_domain(emg_noise, 1000, "EMG_Noise")
