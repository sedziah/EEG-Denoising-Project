import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)
st.write("Model loaded successfully.")

# Function to preprocess the input noisy signal
def preprocess_input(noisy_signal, amplitude, frequency, sampling_rate=1000, duration=1):
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Create harmonic components based on the provided amplitude and frequency
    first_harmonic = (amplitude / 2) * np.sin(2 * np.pi * frequency * t + np.pi/4)
    second_harmonic = (amplitude / 4) * np.sin(2 * np.pi * 2 * frequency * t + np.pi/2)
    
    # Prepare input features by combining the noisy signal with harmonics and phase information
    features = np.column_stack((
        noisy_signal,
        first_harmonic,
        second_harmonic,
        np.full_like(t, np.pi/4),  # Phase of 1st Harmonic
        np.full_like(t, np.pi/2)   # Phase of 2nd Harmonic
    ))
    
    return features

# Function to predict the clean signal from a noisy input
def predict_base_signal(noisy_signal, amplitude, frequency):
    features = preprocess_input(noisy_signal, amplitude, frequency)
    timesteps = features.shape[0]
    X_test = features.reshape(1, timesteps, features.shape[1])  # Single sample input
    predicted_clean_signal = model.predict(X_test)[0]
    return predicted_clean_signal

# Streamlit UI elements
st.title("SSVEP Denoising Model")

# User inputs for amplitude, frequency, and noise level
amplitude = st.slider("Signal Amplitude (µV)", 1.0, 10.0, 5.0)
frequency = st.slider("Signal Frequency (Hz)", 1, 20, 10)
noise_std = st.slider("Noise Level (Standard Deviation)", 0.0, 2.0, 0.5)

# Generate a noisy signal based on user inputs
sampling_rate = 1000
duration = 1  # 1 second
t = np.arange(0, duration, 1/sampling_rate)
clean_signal = amplitude * np.sin(2 * np.pi * frequency * t)
noisy_signal = clean_signal + np.random.normal(0, noise_std, len(t))  # Adding noise

# Predict the clean signal using the model
predicted_clean_signal = predict_base_signal(noisy_signal, amplitude, frequency)

# Plot the results
st.write("### Input Noisy Signal vs. Predicted Clean Signal")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, noisy_signal, label="Input Noisy Signal", linestyle='--')
ax.plot(t, predicted_clean_signal, label="Predicted Clean Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title("Noisy Signal vs. Predicted Clean Signal")
ax.legend()
st.pyplot(fig)
