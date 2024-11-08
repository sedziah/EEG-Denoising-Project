import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_file
import tensorflow as tf
import io
import base64
import os

app = Flask(__name__)

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)

# Function to generate a sine wave and save it as a CSV file
def generate_sine_wave(frequency, amplitude, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    df = pd.DataFrame({'Time': t, 'Signal': signal})
    csv_file = "generated_signal.csv"
    df.to_csv(csv_file, index=False)
    return csv_file

# Function to process noisy signal and predict clean signal
def predict_clean_signal(noisy_signal, frequency=10, harmonics=2):
    # Use the length of the noisy signal for all components to match dimensions
    signal_length = len(noisy_signal)
    t = np.linspace(0, 1, signal_length, endpoint=False)

    # Generate harmonic components to match the noisy signal's structure
    harmonic_components = []
    for i in range(1, harmonics + 1):
        harmonic_amplitude = noisy_signal.max() / (i + 1)  # Scale harmonics
        harmonic_phase = np.pi / (i + 1)
        harmonic_freq = frequency * i
        harmonic = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t + harmonic_phase)
        harmonic_components.append(harmonic)

    # Stack harmonics and prepare the model input
    features = np.column_stack(
        [noisy_signal] + harmonic_components + [np.full_like(t, np.pi / (i + 1)) for i in range(harmonics)]
    )

    # Reshape and predict using the model
    X_test = features.reshape(1, signal_length, features.shape[1])
    predicted_clean_signal = model.predict(X_test)[0]
    return predicted_clean_signal

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Get parameters from the form
        frequency = float(request.form.get('frequency', 10))
        amplitude = float(request.form.get('amplitude', 5))
        sampling_rate = int(request.form.get('sampling_rate', 1000))
        duration = float(request.form.get('duration', 1))

        # Generate the signal CSV file
        csv_file = generate_sine_wave(frequency, amplitude, sampling_rate, duration)
        
        # Serve the generated CSV file for download
        return send_file(csv_file, as_attachment=True)

    return render_template('generate.html')

@app.route('/filter', methods=['GET', 'POST'])
def filter_signal():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(url_for('filter_signal'))

        # Read the uploaded file
        df = pd.read_csv(file)
        noisy_signal = df.iloc[:, 1].values  # Assuming the signal data is in the second column
        
        # Filter the signal using the model
        frequency = float(request.form.get("frequency", 10))
        harmonics = int(request.form.get("harmonics", 2))
        clean_signal = predict_clean_signal(noisy_signal, frequency=frequency, harmonics=harmonics)

        # Prepare the output DataFrame with time and clean signal
        df['Clean Signal'] = clean_signal
        clean_csv = "filtered_signal.csv"
        df.to_csv(clean_csv, index=False)

        # Serve the filtered CSV file for download
        return send_file(clean_csv, as_attachment=True)

    return render_template('filter.html')

if __name__ == '__main__':
    app.run(debug=True)
