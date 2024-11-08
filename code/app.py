# app.py

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
from generate_test_signals import generate_signal_with_harmonics
from clean_signal_prediction import predict_clean_signals

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model_path = "ssvep_denoising_model_with_all_features.keras"
model = tf.keras.models.load_model(model_path)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Get parameters from the form
        frequency = float(request.form.get('frequency'))
        amplitude = float(request.form.get('amplitude'))
        sampling_rate = int(request.form.get('sampling_rate'))
        duration = float(request.form.get('duration'))
        
        # Retrieve harmonic factors based on user selection, with a default fallback
        harmonic_option = request.form.get('harmonic_option', '0.5,0.25')  # Default to '0.5,0.25' if not provided
        harmonic_factors = [float(x.strip()) for x in harmonic_option.split(',')]

        # Generate the signal CSV file
        csv_file = generate_signal_with_harmonics(
            amplitude=amplitude,
            frequency=frequency,
            sampling_rate=sampling_rate,
            duration=duration,
            harmonic_factors=harmonic_factors
        )
        
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

        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Specify the model path
        model_path = "ssvep_denoising_model_with_all_features.keras"

        # Run prediction using the imported `predict_clean_signals` function
        mse, df_with_predictions = predict_clean_signals(
            model_path=model_path,
            dataset_path=file_path,
            noisy_signal_column="Clean Signal with Harmonics",
            clean_signal_column="Clean Signal"
        )

        # Save the DataFrame with predictions to a new CSV
        clean_csv = "filtered_signal.csv"
        df_with_predictions.to_csv(clean_csv, index=False)

        # Serve the filtered CSV file for download
        return send_file(clean_csv, as_attachment=True)

    return render_template('filter.html')

if __name__ == '__main__':
    app.run(debug=True)

