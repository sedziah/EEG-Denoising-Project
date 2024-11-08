import numpy as np
import pandas as pd

def generate_signal_with_harmonics(amplitude, frequency, sampling_rate, duration, harmonic_factors):
    """
    Generate a signal with harmonics based on specified parameters and save it as a CSV file.

    Parameters:
    - amplitude (float): Amplitude of the base signal in µV.
    - frequency (float): Base frequency of the signal in Hz.
    - sampling_rate (int): Sampling rate in samples per second.
    - duration (float): Duration of the signal in seconds.
    - harmonic_factors (list of float): List of scaling factors for each harmonic.
    
    Returns:
    - str: The filename of the generated CSV file.
    """
    # Generate time vector based on duration and sampling rate
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    # Generate the base signal
    base_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Generate harmonics based on harmonic_factors
    harmonics = []
    for i, factor in enumerate(harmonic_factors, start=1):
        harmonic_freq = frequency * (i + 1)  # Each harmonic is a multiple of the base frequency
        phase_shift = (i + 1) * np.pi / 4    # Example phase shift
        harmonic = (amplitude * factor) * np.sin(2 * np.pi * harmonic_freq * t + phase_shift)
        harmonics.append(harmonic)

    # Combine base signal and harmonics to form the noisy signal
    noisy_signal = base_signal + sum(harmonics)

    # Prepare data dictionary for DataFrame
    data = {
        "Time": t,
        "Noisy Signal": noisy_signal,
        "Base Signal": base_signal
    }
    for idx, harmonic in enumerate(harmonics, start=1):
        data[f"Harmonic {idx}"] = harmonic

    # Save data to CSV
    csv_filename = "generated_signal_with_harmonics.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)

    print(f"CSV file '{csv_filename}' generated successfully.")
    return csv_filename
