import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Prepare test data (assuming you have split data previously or use same data loading process)
dataset_path = "frequency_harmonics_dataset_ordered.parquet"
data = pd.read_parquet(dataset_path)

# Select relevant columns for features and target
features = data[[
    "Clean Signal with Harmonics",
    "First Harmonic Amplitude",
    "Second Harmonic Amplitude",
    "Phase of 1st Harmonic",
    "Phase of 2nd Harmonic"
]].values
target = data["Clean Signal"].values

# Reshape data for testing
timesteps = 1000
feature_count = features.shape[1]

X = features.reshape(-1, timesteps, feature_count)
y = target.reshape(-1, timesteps)

# Use the test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate predictions
predictions = model.predict(X_test)

# Calculate Mean Squared Error for evaluation
mse = mean_squared_error(y_test.flatten(), predictions.flatten())
print(f"Test Mean Squared Error: {mse}")

# Plot a comparison of true and predicted signals for a sample
sample_index = 0  # choose a sample to visualize
plt.figure(figsize=(12, 6))
plt.plot(y_test[sample_index], label="True Clean Signal", linestyle='--')
plt.plot(predictions[sample_index], label="Predicted Clean Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title("True vs. Predicted Clean Signal for Sample")
plt.legend()
plt.show()
 