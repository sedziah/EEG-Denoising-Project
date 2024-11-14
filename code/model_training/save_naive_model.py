from code.python_packages import *

# Define the Naive model architecture
def create_naive_model(timesteps, feature_count):
    model = tf.keras.Sequential([
        layers.Input(shape=(timesteps, feature_count)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(timesteps, activation="linear")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9),
        loss="mean_squared_error"
    )
    return model

# Define parameters for initialization
timesteps = 1000  # Use same value as LSTM model
feature_count = 5  # Number of features expected by model

# Initialize the Naive model without training
naive_model = create_naive_model(timesteps, feature_count)

# Save the untrained Naive model
naive_model.save("ssvep_denoising_model_naive.keras")
print("Naive model saved as ssvep_denoising_model_naive.keras")
