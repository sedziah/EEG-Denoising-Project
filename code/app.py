import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import io
import base64

app = Flask(__name__)

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)

# Function to process noisy signal and predict clean signal
def predict_clean_signal(noisy_signal, frequency=10, harmonics=2):
    sampling_rate = 1000
    t = np.arange(0, 1, 1 / sampling_rate)
    
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

    # Predict using the model
    X_test = features.reshape(1, len(t), features.shape[1])
    predicted_clean_signal = model.predict(X_test)[0]
    return predicted_clean_signal

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST', 'GET'])
def extract():
    # Redirect GET requests to the index page
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    # Process POST request
    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(url_for('index'))

    # Read the uploaded file into a DataFrame
    df = pd.read_csv(file)
    noisy_signal = df.iloc[:, 0].values  # Assume first column contains the noisy signal
    
    # Retrieve additional parameters
    frequency = float(request.form.get("frequency", 10))
    harmonics = int(request.form.get("harmonics", 2))
    
    # Predict clean signal using the model
    predicted_clean_signal = predict_clean_signal(noisy_signal, frequency=frequency, harmonics=harmonics)

    # Plot and save the figure as a base64 image
    plt.figure(figsize=(12, 6))
    t = np.linspace(0, 1, len(noisy_signal))
    plt.plot(t, noisy_signal, label="Input Noisy Signal", linestyle='--')
    plt.plot(t, predicted_clean_signal, label="Predicted Clean Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Noisy Signal vs. Predicted Clean Signal")
    plt.legend()

    # Convert plot to PNG image in base64 for embedding
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
