# main.py

from signal_simulator import (
    generate_clean_eeg_signal,
    generate_eog_noise,
    generate_emg_noise,
    plot_signals,
)

# Generate signals
clean_signal = generate_clean_eeg_signal()
eog_noise = generate_eog_noise(-3)  # Adjust SNR as needed
emg_noise = generate_emg_noise(-5)  # Adjust SNR as needed

# Plot the signals
plot_signals(clean_signal, eog_noise, emg_noise)
