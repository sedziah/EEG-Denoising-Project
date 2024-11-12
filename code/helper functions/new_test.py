import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Generate synthetic noisy data with superimposed harmonics
def generate_test_data(amplitude=5, frequency=10, sampling_rate=1000, duration=1, noise_std=0.5):
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Generate the clean base signal at 10 Hz with amplitude 5 µV
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Generate first harmonic (same frequency as the fundamental with a phase shift)
    first_harmonic = (amplitude / 2) * np.sin(2 * np.pi * frequency * t + np.pi/4)
    
    # Generate second harmonic (double the frequency of the fundamental)
    second_harmonic = (amplitude / 4) * np.sin(2 * np.pi * 2 * frequency * t + np.pi/2)
    
    # Combine the fundamental and harmonic components to create a noisy signal
    clean_signal_with_harmonics = clean_signal + first_harmonic + second_harmonic
    
    # Add Gaussian noise to the combined signal
    noisy_signal = clean_signal_with_harmonics + np.random.normal(0, noise_std, len(t))
    
    # Prepare the model input by stacking all relevant features
    features = np.column_stack((
        noisy_signal,
        first_harmonic,
        second_harmonic,
        np.full_like(t, np.pi/4),  # Phase of 1st Harmonic
        np.full_like(t, np.pi/2)   # Phase of 2nd Harmonic
    ))
    
    return features, clean_signal  # Return both the noisy features and the true clean signal for comparison

# Generate test data
features, true_clean_signal = generate_test_data()

# Reshape for model prediction
timesteps = features.shape[0]
X_test = features.reshape(1, timesteps, features.shape[1])  # Single sample input

# Predict the clean signal using the model
predicted_clean_signal = model.predict(X_test)[0]

# Plot the true and predicted clean signals
t = np.linspace(0, 1, timesteps)
plt.figure(figsize=(12, 6))
plt.plot(t, true_clean_signal, label="True Clean Signal", linestyle='--')
plt.plot(t, predicted_clean_signal, label="Predicted Clean Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("True Clean Signal vs. Predicted Clean Signal")
plt.legend()
plt.show()
