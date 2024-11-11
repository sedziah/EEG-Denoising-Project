import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Parameters for sine wave generation
fundamental_frequency = 10  # 10 Hz base frequency
sampling_rate = 1000        # Sampling rate in Hz
duration = 1                # Duration in seconds
t = np.arange(0, duration, 1 / sampling_rate)

# Amplitude ranges for harmonics
amplitude_fundamental = 5   # Constant amplitude for fundamental
amplitude_1st_harmonic = np.linspace(0, 5, 6)  # Vary from 0 to 5 µV
amplitude_2nd_harmonic = np.linspace(0, 2.5, 6)  # Vary from 0 to 2.5 µV

# Phase ranges for harmonics
phase_values = np.linspace(0, 2 * np.pi, 8)  # Phase steps from 0 to 2π

# Generate dataset
data_records = []

# Loop over all combinations of amplitudes and phases for first and second harmonics
for amp1 in amplitude_1st_harmonic:
    for amp2 in amplitude_2nd_harmonic:
        for phase1 in phase_values:
            for phase2 in phase_values:
                # Generate each component with the specified parameters
                clean_signal = amplitude_fundamental * np.sin(2 * np.pi * fundamental_frequency * t)
                first_harmonic = amp1 * np.sin(2 * np.pi * fundamental_frequency * t + phase1)
                second_harmonic = amp2 * np.sin(2 * np.pi * 2 * fundamental_frequency * t + phase2)
                
                # Combine components to create the clean signal with harmonics
                clean_signal_with_harmonics = clean_signal + first_harmonic + second_harmonic
                
                # Store each sample as a record in the specified order
                for i, time_point in enumerate(t):
                    data_records.append({
                        "Time (s)": time_point,
                        "Clean Signal": clean_signal[i],
                        "First Harmonic Amplitude": first_harmonic[i],
                        "Second Harmonic Amplitude": second_harmonic[i],
                        "Amplitude of 1st Harmonic": amp1,
                        "Amplitude of 2nd Harmonic": amp2,
                        "Phase of 1st Harmonic": phase1,
                        "Phase of 2nd Harmonic": phase2,
                        "Clean Signal with Harmonics": clean_signal_with_harmonics[i]
                    })

# Convert records to a DataFrame
dataset_df = pd.DataFrame(data_records)

# Save the dataset to a Parquet file
dataset_path = "frequency_harmonics_dataset_ordered.parquet"
table = pa.Table.from_pandas(dataset_df)
pq.write_table(table, dataset_path)

print(f"Dataset saved to {dataset_path} in Parquet format.")