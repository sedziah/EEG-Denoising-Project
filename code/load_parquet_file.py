import pandas as pd

# Load the Parquet file
dataset_path = "frequency_harmonics_dataset_ordered.parquet"  # Updated file name
data = pd.read_parquet(dataset_path)

# Display the first 10 rows of the loaded data
print(data.head(10000))
