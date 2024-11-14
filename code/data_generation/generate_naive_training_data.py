import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# Parameters for the data
sampling_rate = 1000  # Sampling rate per millisecond
signal_duration = 250  # 250 ms duration

# Generate the time range for 250 ms with the correct sampling rate
time = np.arange(0, signal_duration, 1 / sampling_rate)  # from 0 to 250 ms, with 1000 samples per ms

# Create a DataFrame with the specified structure, all zero values except for time
df_naive = pd.DataFrame({
    'Time': time,
    'Clean Signal': np.zeros(len(time)),
    'Second Harmonic Amplitude': np.zeros(len(time)),
    'Second Harmonic Phase': np.zeros(len(time)),
    'Third Harmonic Amplitude': np.zeros(len(time)),
    'Third Harmonic Phase': np.zeros(len(time)),
    'Noisy Signal': np.zeros(len(time))
})

# Save to Parquet file in the current directory
parquet_table = pa.Table.from_pandas(df_naive)
pq.write_table(parquet_table, 'naive_model_training_data.parquet')

# Display the first few rows of the DataFrame for confirmation
print(df_naive.head())
