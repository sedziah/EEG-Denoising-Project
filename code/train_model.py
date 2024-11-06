# train_model.py

import numpy as np
import pandas as pd
from network_structure import RNN_lstm  # Import your LSTM model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from signal_simulator import generate_clean_eeg_signal, generate_eog_noise, generate_emg_noise, save_signals_to_csv
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
raw_data_path = "../data/mean_clean_eeg_and_noise_time_domain.csv"
save_dir = "../data/split_data"
train_path = os.path.join(save_dir, "train_data.csv")
val_path = os.path.join(save_dir, "val_data.csv")
test_path = os.path.join(save_dir, "test_data.csv")
std_dev_path = os.path.join(save_dir, "std_composite.txt")
os.makedirs(save_dir, exist_ok=True)

def ensure_data_exists():
    """Generate raw signal data if it doesn't already exist."""
    if not Path(raw_data_path).exists():
        print("Generating raw signals...")
        clean_signal = generate_clean_eeg_signal()
        eog_noise = generate_eog_noise(-3)
        emg_noise = generate_emg_noise(-5)
        save_signals_to_csv(clean_signal, eog_noise, emg_noise)
    else:
        print("Raw signal data already exists.")

def load_signals():
    """Loads the clean EEG, EOG noise, EMG noise, and composite signal from a CSV file."""
    df = pd.read_csv(raw_data_path)
    signals = {
        "time": df["Time (s)"].to_numpy(),
        "clean_eeg": df["Clean EEG Signal (µV)"].to_numpy() * 1e-6,  # Convert to volts
        "eog_noise": df["EOG Noise (µV)"].to_numpy() * 1e-6,
        "emg_noise": df["EMG Noise (µV)"].to_numpy() * 1e-6,
        "composite_signal": df["Composite Signal (µV)"].to_numpy() * 1e-6
    }
    return signals

def normalize_signals(signals):
    """Normalizes signals based on composite signal's standard deviation."""
    composite_signal = signals["composite_signal"]
    clean_eeg = signals["clean_eeg"]
    eog_noise = signals["eog_noise"]
    emg_noise = signals["emg_noise"]

    std_composite = np.std(composite_signal)
    normalized_signals = {
        "normalized_composite": composite_signal / std_composite,
        "normalized_clean": clean_eeg / std_composite,
        "normalized_eog": eog_noise / std_composite,
        "normalized_emg": emg_noise / std_composite,
        "std_composite": std_composite
    }
    return normalized_signals

def split_data(normalized_data, train_ratio=0.7, val_ratio=0.15):
    """Splits normalized data into training, validation, and test sets."""
    X = normalized_data["normalized_composite"]
    y = normalized_data["normalized_clean"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_ratio, random_state=42)
    test_ratio = 1 - (train_ratio + val_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

def save_split_data(data_splits):
    """Saves split data into separate CSV files for training, validation, and test sets."""
    for split_name, (X, y) in data_splits.items():
        split_data = {
            "Normalized Composite Signal": X,
            "Normalized Clean EEG Signal": y,
        }
        file_path = os.path.join(save_dir, f"{split_name}_data.csv")
        pd.DataFrame(split_data).to_csv(file_path, index=False)
        print(f"{split_name.capitalize()} data saved to {file_path}")

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
    for i in range(len(X_composite) - timesteps):
        # Stack all three features into the input sequence
        X_seq.append(np.stack([X_composite[i:i + timesteps], X_eog[i:i + timesteps], X_emg[i:i + timesteps]], axis=-1))
        y_seq.append(y[i:i + timesteps])  # Target is still the clean EEG signal
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    return X_seq, y_seq

if __name__ == "__main__":
    # Ensure raw signal data exists
    ensure_data_exists()

    # Load raw signals and preprocess if needed
    signals = load_signals()
    normalized_data = normalize_signals(signals)

    # Save standard deviation for denormalization
    with open(std_dev_path, "w") as f:
        f.write(str(normalized_data["std_composite"]))
        print(f"Standard deviation saved to {std_dev_path}")

    # Split and save data if not already done
    data_splits = split_data(normalized_data)
    save_split_data(data_splits)

    # Load preprocessed data
    X_train, y_train = load_preprocessed_data(train_path)
    X_val, y_val = load_preprocessed_data(val_path)

    # Check if data loaded correctly
    if X_train is not None and X_val is not None:
        timesteps = 10  # Number of timesteps for LSTM input

        # Prepare data for LSTM using all features
        X_train, y_train = prepare_data(normalized_data["normalized_composite"], 
                                        normalized_data["normalized_eog"], 
                                        normalized_data["normalized_emg"], 
                                        y_train, timesteps)
        
        X_val, y_val = prepare_data(normalized_data["normalized_composite"], 
                                    normalized_data["normalized_eog"], 
                                    normalized_data["normalized_emg"], 
                                    y_val, timesteps)

        # Create the model with multiple features
        feature_count = 3  # Composite Signal, EOG Noise, EMG Noise
        model = RNN_lstm(timesteps, feature_count=feature_count)
        
        # Compile the model with a custom optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Define model checkpoint callback
        checkpoint_path = "../data/best_lstm_model.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')

        # Train the model with validation data
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

        # Save the final trained model
        final_model_path = "../data/final_trained_lstm_model.keras"
        model.save(final_model_path)
        print(f"Model trained and saved as {final_model_path}")
    else:
        print("Training aborted due to data loading failure.")
