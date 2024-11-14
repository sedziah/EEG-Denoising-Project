from code.model_training.python_packages import *


def generate_signal_with_harmonics(amplitude, frequency, sampling_rate, duration, harmonic_factors=[0.5, 0.25]):
    """
    Generate a signal with harmonics based on specified parameters and save it as a CSV file with values rounded to 2 decimal places.

    Parameters:
    - amplitude (float): Amplitude of the base signal in µV.
    - frequency (float): Base frequency of the signal in Hz.
    - sampling_rate (int): Sampling rate in samples per second.
    - duration (float): Duration of the signal in seconds.
    - harmonic_factors (list of float): List of scaling factors for each harmonic.
    
    Returns:
    - str: The filename of the generated CSV file.
    """
    # Generate time vector using np.linspace for precise interval spacing
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    # Generate the base (clean) signal
    base_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    # Generate harmonics based on harmonic_factors
    harmonics = []
    for i, factor in enumerate(harmonic_factors, start=1):
        harmonic_freq = frequency * (i + 1)  # Each harmonic is a multiple of the base frequency
        phase_shift = (i + 1) * np.pi / 4    # Example phase shift
        harmonic = (amplitude * factor) * np.sin(2 * np.pi * harmonic_freq * t + phase_shift)
        harmonics.append(harmonic)

    # Sum the base signal and harmonics to create the noisy signal
    noisy_signal = base_signal + sum(harmonics)

    # Prepare data dictionary for DataFrame in the specified order
    data = {
        "Time": t,
        "Clean Signal": base_signal
    }
    for idx, harmonic in enumerate(harmonics, start=1):
        data[f"Harmonic {idx}"] = harmonic
    data["Noisy Signal"] = noisy_signal  # Add noisy signal as the last column

    # Create DataFrame and round values to 2 decimal places
    df = pd.DataFrame(data).round(2)

    # Save data to CSV
    csv_filename = "generated_signal_with_harmonics.csv"
    df.to_csv(csv_filename, index=False)

    print(f"CSV file '{csv_filename}' generated successfully with values rounded to 2 decimal places.")
    return csv_filename

# Test the function with sample parameters
generate_signal_with_harmonics(
    amplitude=5,          # Amplitude of 5 µV
    frequency=7.5,         # Base frequency of 10 Hz
    sampling_rate=1000,   # Sampling rate of 100 samples per second
    duration=2,           # Duration of 2 seconds
    harmonic_factors=[0.5, 0.25]  # Scaling factors for harmonics
)
