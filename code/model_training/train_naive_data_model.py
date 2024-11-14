import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load the Parquet file
dataset_path = "naive_model_training_data.parquet"
data = pd.read_parquet(dataset_path)

# Prepare data for training
# Use "Second Harmonic Amplitude", "Second Harmonic Phase", "Third Harmonic Amplitude", "Third Harmonic Phase", "Noisy Signal" as features
# Target is the "Clean Signal"

# Select all relevant columns for input features
features = data[[
    "Second Harmonic Amplitude",
    "Second Harmonic Phase",
    "Third Harmonic Amplitude",
    "Third Harmonic Phase",
    "Noisy Signal"
]].values
target = data["Clean Signal"].values  # The pure fundamental frequency without harmonics

# Reshape data for LSTM input format: (samples, timesteps, features)
timesteps = 250  # Based on the 250 ms duration
feature_count = features.shape[1]

# Reshape features and target into 3D and 2D arrays respectively
X = features.reshape(-1, timesteps, feature_count)
y = target.reshape(-1, timesteps, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
def create_naive_model(timesteps, feature_count):
    model = tf.keras.Sequential([
        layers.Input(shape=(timesteps, feature_count)),
        layers.LSTM(16, return_sequences=True),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear")  # Predicting the clean amplitude per timestep
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error"
    )
    return model

# Initialize and train the model
model = create_naive_model(timesteps, feature_count)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,  # Using fewer epochs for a naive model
    batch_size=4,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Save the model in .keras format
model.save("naive_model.keras")
print("Model saved as naive_model.keras")
