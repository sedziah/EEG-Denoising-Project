# train_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from signal_simulator import (
    generate_clean_eeg_signal,
    generate_eog_noise,
    generate_emg_noise,
    save_signals_to_csv,
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
raw_data_path = "../data/mean_clean_eeg_and_noise_with_snr_time_domain.csv"
save_dir = "../data/split_data"
train_path = os.path.join(save_dir, "train_data.csv")
val_path = os.path.join(save_dir, "val_data.csv")
test_path = os.path.join(save_dir, "test_data.csv")
std_dev_path = os.path.join(save_dir, "std_composite.txt")
os.makedirs(save_dir, exist_ok=True)


   


def load_signals():
    """Loads the clean EEG, EOG noise, EMG noise, and composite signal from a CSV file."""
    df = pd.read_csv(raw_data_path)
    signals = {
        "time": df["Time (s)"].to_numpy(),
        "clean_eeg": df["Clean EEG Signal (µV)"].to_numpy() * 1e-6,  # Convert to volts
        "eog_noise": df["EOG Noise (µV)"].to_numpy() * 1e-6,
        "emg_noise": df["EMG Noise (µV)"].to_numpy() * 1e-6,
        "composite_signal": df["Composite Signal (µV)"].to_numpy() * 1e-6,
    }
    return signals


def normalize_signals(signals):
    """Normalizes signals based on composite signal's standard deviation."""
    composite_signal = signals["composite_signal"]
    std_composite = np.std(composite_signal)
    return {k: v / std_composite for k, v in signals.items()}, std_composite


def split_data(normalized_data, train_ratio=0.7, val_ratio=0.15):
    """Splits normalized data into training, validation, and test sets with all features."""

    # Extract individual features
    X_composite = normalized_data["composite_signal"]
    X_eog = normalized_data["eog_noise"]
    X_emg = normalized_data["emg_noise"]
    y = normalized_data["clean_eeg"]

    # Stack features into a single array for splitting (shape: (samples, 3))
    X = np.stack([X_composite, X_eog, X_emg], axis=-1)

    # First, split into train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )

    # Calculate the test ratio based on remaining data
    test_ratio = 1 - (train_ratio + val_ratio)

    # Split the temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
    )

    # Return as a dictionary of splits
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def save_split_data(data_splits, save_dir):
    """Saves split data into separate CSV files for training, validation, and test sets."""
    os.makedirs(save_dir, exist_ok=True)
    for split_name, (X, y) in data_splits.items():
        # Save as a DataFrame with separate columns for each feature and the target
        df = pd.DataFrame(
            {
                "Normalized Composite Signal": X[:, 0],
                "Normalized EOG Noise": X[:, 1],
                "Normalized EMG Noise": X[:, 2],
                "Normalized Clean EEG Signal": y,
            }
        )
        df.to_csv(os.path.join(save_dir, f"{split_name}_data.csv"), index=False)
        print(
            f"{split_name.capitalize()} data saved to {os.path.join(save_dir, f'{split_name}_data.csv')}"
        )


def prepare_data(X_composite, X_eog, X_emg, y, timesteps):
    """
    Prepares the data for training by creating input sequences with multiple features.

    Args:
        X_composite (np.ndarray): Normalized composite signal.
        X_eog (np.ndarray): Normalized EOG noise.
        X_emg (np.ndarray): Normalized EMG noise.
        y (np.ndarray): Normalized clean signal (target).
        timesteps (int): Number of timesteps for LSTM input.

    Returns:
        tuple: (X, y) where X is the input data reshaped for LSTM, and y is the target reshaped for sequence output.
    """
    X_seq, y_seq = [], []

    # Loop only up to len(y) - timesteps to avoid out-of-bounds access
    for i in range(len(y) - timesteps):
        # Create sequences for each feature
        composite_seq = X_composite[i : i + timesteps]
        eog_seq = X_eog[i : i + timesteps]
        emg_seq = X_emg[i : i + timesteps]

        # Ensure each sequence has the correct length before stacking
        if (
            len(composite_seq) == timesteps
            and len(eog_seq) == timesteps
            and len(emg_seq) == timesteps
        ):
            X_seq.append(np.stack([composite_seq, eog_seq, emg_seq], axis=-1))
            y_seq.append(
                y[i + timesteps - 1]
            )  # Use only the last value in the sequence as target

    # Convert lists to numpy arrays
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq


def load_preprocessed_data(file_path):
    """Loads preprocessed time-domain data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        X = df["Normalized Composite Signal"].to_numpy()
        y = df["Normalized Clean EEG Signal"].to_numpy()
        return X, y
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None


if __name__ == "__main__":
    # Ensure raw signal data exists
    ensure_data_exists()

    # Load and normalize signals
    signals = load_signals()
    normalized_data, std_composite = normalize_signals(signals)

    # Save standard deviation for later use
    with open(std_dev_path, "w") as f:
        f.write(str(std_composite))

    # Split and save data
    data_splits = split_data(normalized_data)
    save_split_data(data_splits)

    # Load data for model
    X_train, y_train = data_splits["train"]
    timesteps = 10

    # Prepare LSTM data
    X_train, y_train = prepare_data(
        normalized_data["composite_signal"],
        normalized_data["eog_noise"],
        normalized_data["emg_noise"],
        y_train,
        timesteps,
    )

    print("Data prepared and ready for model training.")
