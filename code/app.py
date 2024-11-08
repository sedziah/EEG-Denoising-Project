from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)

# Global variable to store generated data for extraction
generated_data = None
true_clean_signal = None

# Function to generate synthetic noisy data with harmonics
def generate_data(amplitude=5, frequency=10, harmonics=2, sampling_rate=1000, duration=1, noise_std=0.5):
    t = np.arange(0, duration, 1/sampling_rate)
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Generate harmonics based on the number of harmonics specified
    harmonic_components = []
    for i in range(1, harmonics + 1):
        harmonic_amplitude = amplitude / (i + 1)
        harmonic_phase = np.pi / (i + 1)
        harmonic_freq = frequency * i
        harmonic = harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * t + harmonic_phase)
        harmonic_components.append(harmonic)
    
    # Combine all harmonics to create the noisy signal
    noisy_signal = clean_signal + sum(harmonic_components) + np.random.normal(0, noise_std, len(t))
    
    # Prepare input features
    global generated_data, true_clean_signal
    generated_data = np.column_stack(
        [noisy_signal] + harmonic_components + [np.full_like(t, np.pi / (i + 1)) for i in range(harmonics)]
    )
    true_clean_signal = clean_signal

# Function to extract clean data using the model
def extract_clean_data():
    if generated_data is None:
        return None, None  # No data generated yet
    
    timesteps = generated_data.shape[0]
    X_test = generated_data.reshape(1, timesteps, generated_data.shape[1])
    predicted_clean_signal = model.predict(X_test)[0]
    return true_clean_signal, predicted_clean_signal

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    frequency = float(request.form.get("frequency", 10))
    harmonics = int(request.form.get("harmonics", 2))
    generate_data(frequency=frequency, harmonics=harmonics)
    return redirect(url_for('index'))

@app.route('/extract', methods=['POST'])
def extract():
    true_signal, predicted_signal = extract_clean_data()
    if true_signal is None or predicted_signal is None:
        return redirect(url_for('index'))
    
    # Plot and save the figure as a base64 image
    plt.figure(figsize=(12, 6))
    t = np.linspace(0, 1, len(true_signal))
    plt.plot(t, true_signal, label="True Clean Signal", linestyle='--')
    plt.plot(t, predicted_signal, label="Predicted Clean Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("True vs. Predicted Clean Signal")
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
