# main.py

from train_model import (
    ensure_data_exists,
    load_signals,
    normalize_signals,
    split_data,
    save_split_data,
    load_preprocessed_data,
    prepare_data,
)
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from network_structure import RNN_lstm

# Define paths for the data files
train_path = "../data/split_data/train_data.csv"
val_path = "../data/split_data/val_data.csv"
std_dev_path = "../data/split_data/std_composite.txt"
save_dir = "../data/split_data"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    # Ensure that raw signal data exists
    ensure_data_exists()

    # Load and normalize signals
    signals = load_signals()
    normalized_data, std_composite = normalize_signals(signals)

    # Save standard deviation for later use
    with open(std_dev_path, "w") as f:
        f.write(str(std_composite))
        print(f"Standard deviation saved to {std_dev_path}")

    # Split and save data if not already done
    data_splits = split_data(normalized_data)
    save_split_data(data_splits, save_dir)

    # Load preprocessed training and validation data
    X_train, y_train = load_preprocessed_data(train_path)
    X_val, y_val = load_preprocessed_data(val_path)

    # Check if data loaded correctly
    if X_train is not None and X_val is not None:
        timesteps = 10  # Set the number of timesteps for LSTM input

        # Prepare data for LSTM model
        X_train, y_train = prepare_data(
            normalized_data["composite_signal"],
            normalized_data["eog_noise"],
            normalized_data["emg_noise"],
            y_train,
            timesteps,
        )
        X_val, y_val = prepare_data(
            normalized_data["composite_signal"],
            normalized_data["eog_noise"],
            normalized_data["emg_noise"],
            y_val,
            timesteps,
        )

        # Define feature count for the LSTM input (3: composite signal, EOG, and EMG noise)
        feature_count = 3
        model = RNN_lstm(timesteps, feature_count=feature_count)

        # Compile model with specified optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9)
        model.compile(optimizer=optimizer, loss="mse")

        # Define model checkpoint to save best weights
        checkpoint_path = "../data/best_lstm_model.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_loss"
        )

        # Train the model and capture the history
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],
        )

        # Save the final trained model
        final_model_path = "../data/final_trained_lstm_model.keras"
        model.save(final_model_path)
        print(f"Model trained and saved as {final_model_path}")

        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()

        # Save the plot
        loss_plot_path = "../data/loss_plot.png"
        plt.savefig(loss_plot_path)
        print(f"Loss plot saved to {loss_plot_path}")

        plt.show()

    else:
        print("Training aborted due to data loading failure.")
