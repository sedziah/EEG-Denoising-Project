# main.py

from train_model import ensure_data_exists, load_signals, normalize_signals, split_data, save_split_data, load_preprocessed_data, prepare_data, RNN_lstm
import tensorflow as tf
import os

# Paths to training, validation, and test data
train_path = "../data/split_data/train_data.csv"
val_path = "../data/split_data/val_data.csv"
std_dev_path = "../data/split_data/std_composite.txt"

# Ensure data exists and is preprocessed
ensure_data_exists()  # This will generate raw data if it doesn’t already exist

# Load and preprocess data
signals = load_signals()
normalized_data = normalize_signals(signals)
data_splits = split_data(normalized_data)
save_split_data(data_splits)

# Load preprocessed data
X_train, y_train = load_preprocessed_data(train_path)
X_val, y_val = load_preprocessed_data(val_path)

# Check if data loaded correctly
if X_train is not None and X_val is not None:
    timesteps = 10  # Number of timesteps for LSTM input

    # Prepare data for LSTM
    X_train, y_train = prepare_data(X_train, y_train, timesteps)
    X_val, y_val = prepare_data(X_val, y_val, timesteps)

    # Create the model
    model = RNN_lstm(timesteps)
    model.summary()  # Print model summary

    # Compile the model with custom optimizer
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
