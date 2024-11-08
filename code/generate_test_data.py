import numpy as np
import pandas as pd

# Parameters for the signal
amplitude = 5           # Amplitude of 5 µV for the base signal
frequency = 10          # Base frequency of 10 Hz
sampling_rate = 100    # Sampling rate of 1000 samples per second
duration = 2            # Duration of the signal in seconds
second_harmonic_freq = 2 * frequency  # Frequency of the second harmonic (20 Hz)

# Generate a single time vector for 1 second at 1000 Hz
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the base signal (10 Hz)
base_signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Generate the first harmonic (10 Hz with phase shift)
first_harmonic = (amplitude / 2) * np.sin(2 * np.pi * frequency * t + np.pi / 4)

# Generate the second harmonic (20 Hz with phase shift)
second_harmonic = (amplitude / 4) * np.sin(2 * np.pi * second_harmonic_freq * t + np.pi / 2)

# Combine the base signal and harmonics to create the noisy signal
noisy_signal = base_signal + first_harmonic + second_harmonic

# Save the data to a CSV file
data = {
    "Noisy Signal": noisy_signal,
    "Base Signal": base_signal,
    "First Harmonic": first_harmonic,
    "Second Harmonic": second_harmonic
}
df = pd.DataFrame(data)
df.to_csv("noisy_signal_with_harmonics.csv", index=False)

print("CSV file 'noisy_signal_with_harmonics.csv' generated successfully.")
