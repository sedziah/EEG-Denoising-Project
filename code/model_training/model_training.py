from python_packages import *


# Load the Parquet file
dataset_path = "frequency_harmonics_dataset_ordered.parquet"
data = pd.read_parquet(dataset_path)

# Prepare data for training
# Use "Clean Signal with Harmonics" (noisy data), "First Harmonic Amplitude", "Second Harmonic Amplitude",
# "Phase of 1st Harmonic", "Phase of 2nd Harmonic" as features
# Target is the "Clean Signal" (base signal)

# Select all relevant columns for input features
features = data[[
    "Clean Signal with Harmonics",
    "First Harmonic Amplitude",
    "Second Harmonic Amplitude",
    "Phase of 1st Harmonic",
    "Phase of 2nd Harmonic"
]].values
target = data["Clean Signal"].values  # The pure fundamental frequency without harmonics

# Reshape data for LSTM input format: (samples, timesteps, features)
timesteps = 1000  # Assuming each sequence is 1 second at 1000 Hz
feature_count = features.shape[1]

# Reshape features and target into 3D and 2D arrays respectively
X = features.reshape(-1, timesteps, feature_count)
y = target.reshape(-1, timesteps)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
def create_model(timesteps, feature_count):
    model = tf.keras.Sequential([
        layers.Input(shape=(timesteps, feature_count)),
        layers.LSTM(64, return_sequences=True),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="linear")  # Predicting the clean amplitude per timestep
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9),
        loss="mean_squared_error"
    )
    return model

# Initialize and train the model
model = create_model(timesteps, feature_count)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Save the model in .keras format
model.save("ssvep_denoising_model_with_all_features.keras")
print("Model saved as ssvep_denoising_model_with_all_features.keras")
