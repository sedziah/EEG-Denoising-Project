# test_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the trained model
model_path = "../data/final_trained_lstm_model.keras"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Load the test data
test_data_path = "../data/split_data/test_data.csv"
std_dev_path = "../data/split_data/std_composite.txt"

# Load the standard deviation for denormalization, if needed
try:
    with open(std_dev_path, "r") as f:
        std_dev = float(f.read().strip())
        print(f"Standard deviation loaded: {std_dev}")
except FileNotFoundError:
    std_dev = None
    print(f"Standard deviation file not found: {std_dev_path}")

# Load and prepare test data
test_df = pd.read_csv(test_data_path)
X_test_composite = test_df["Normalized Composite Signal"].to_numpy()
X_test_eog = test_df["Normalized EOG Noise"].to_numpy()
X_test_emg = test_df["Normalized EMG Noise"].to_numpy()
y_test = test_df["Normalized Clean EEG Signal"].to_numpy()

# Reshape for LSTM input
timesteps = 10  # Must match the timesteps used during training
X_test_seq, y_test_seq = [], []

# Create sequences for testing
for i in range(len(X_test_composite) - timesteps):
    # Stack all three features into the input sequence
    composite_seq = X_test_composite[i:i + timesteps]
    eog_seq = X_test_eog[i:i + timesteps]
    emg_seq = X_test_emg[i:i + timesteps]
    X_test_seq.append(np.stack([composite_seq, eog_seq, emg_seq], axis=-1))  # Shape (timesteps, 3)
    y_test_seq.append(y_test[i + timesteps - 1])  # Target is the last value of each sequence

X_test_seq = np.array(X_test_seq)  # Shape: (num_samples, timesteps, 3)
y_test_seq = np.array(y_test_seq)  # Shape: (num_samples,)

# Run predictions
y_pred_seq = model.predict(X_test_seq)

# Denormalize predictions and true values if standard deviation is available
if std_dev:
    y_test_seq = y_test_seq * std_dev
    y_pred_seq = y_pred_seq * std_dev

# Calculate MSE for predictions
mse = mean_squared_error(y_test_seq, y_pred_seq)
print(f"Mean Squared Error: {mse}")

# Plotting the expected vs. predicted results
plt.figure(figsize=(12, 6))
plt.plot(y_test_seq[:100], label="Expected (True Values)", color="blue")  # Plot first 100 samples for clarity
plt.plot(y_pred_seq[:100], label="Predicted Values", color="red", linestyle="dashed")
plt.title("Predicted vs. Expected Results (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.show()

# Optional: Save results to a CSV file for further analysis
results_df = pd.DataFrame({"Expected": y_test_seq.flatten(), "Predicted": y_pred_seq.flatten()})
results_df.to_csv("../data/predicted_vs_expected.csv", index=False)
print("Predicted vs. expected results saved to ../data/predicted_vs_expected.csv")
