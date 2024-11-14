from code.python_packages import *
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Define paths to model and dataset
model_path = "ssvep_denoising_model_naive.keras"
dataset_path = "generated_signal_with_harmonics.csv"
output_path = "predicted_signals_with_harmonics.csv"  # Define output path for the new CSV file

def predict_clean_signals(
    model_path, 
    dataset_path, 
    noisy_signal_column="Noisy Signal", 
    clean_signal_column="Clean Signal"
):
    """
    Evaluates a trained model on the entire dataset as a single sequence.

    Parameters:
    - model_path (str): Path to the trained model (.keras file).
    - dataset_path (str): Path to the dataset file (.csv).
    - noisy_signal_column (str): Column name for the noisy signal (input to the model).
    - clean_signal_column (str): Column name for the clean signal (ground truth).

    Returns:
    - mse (float): Mean Squared Error between predicted and actual clean signal.
    - data (DataFrame): Original DataFrame with an added column for predicted clean signal and accuracy.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    print("Model input shape:", model.input_shape)  # Check the model's expected input shape

    # Load the dataset
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully with columns:", data.columns)

    # Verify that the specified columns exist in the dataset
    if noisy_signal_column not in data.columns or clean_signal_column not in data.columns:
        raise ValueError(f"Columns '{noisy_signal_column}' or '{clean_signal_column}' not found in the dataset.")

    # Extract the noisy signal as the input feature and the clean signal as the target
    noisy_signal = data[[noisy_signal_column]].values  # Shape: (total_timesteps, 1)
    clean_signal = data[clean_signal_column].values    # Shape: (total_timesteps,)

    # Add dummy features to match the model's expected input shape (total_timesteps, 5)
    dummy_features = np.zeros((noisy_signal.shape[0], 4))  # Create 4 dummy columns
    model_input = np.concatenate([noisy_signal, dummy_features], axis=1)  # Combine into (total_timesteps, 5)

    # Flatten input to exactly match model's expected input shape
    if model.input_shape[-1] == 5000:
        # Ensure the model input has exactly 5000 features
        total_elements_needed = 5000
        if model_input.size < total_elements_needed:
            raise ValueError("The input data is smaller than the required 5000 features for the model.")
        
        # Flatten and truncate or pad to 5000 elements
        flat_input = model_input.flatten()
        if flat_input.size >= total_elements_needed:
            X = flat_input[:total_elements_needed].reshape(1, 5000)  # Take the first 5000 elements and reshape
        else:
            # Pad with zeros if there are fewer than 5000 elements
            X = np.pad(flat_input, (0, total_elements_needed - flat_input.size)).reshape(1, 5000)
    else:
        # Reshape for sequence models if applicable
        X = model_input.reshape(1, -1, 5)  # Shape to (1, total_timesteps, 5)

    y = clean_signal.reshape(1, -1)    # Shape to (1, total_timesteps)

    # Generate predictions on the entire dataset using direct model call for single input
    predictions = model(X, training=False).numpy().flatten()

    # Calculate Mean Squared Error for evaluation
    mse = mean_squared_error(y.flatten(), predictions[:y.size])
    print(f"Test Mean Squared Error: {mse}")

    # Add predictions as a new column in the original DataFrame
    data['Predicted Clean Signal'] = predictions[:len(data)]

    # Calculate percentage accuracy and add it as a new column
    epsilon = 1e-8  # Small constant to avoid division by zero
    data['Accuracy (%)'] = np.minimum(
        100, 100 * (1 - np.abs(data[clean_signal_column] - data['Predicted Clean Signal']) / np.maximum(np.abs(data[clean_signal_column]), epsilon))
    )

    # Round all values to 2 decimal places before saving
    data = data.round(2)

    # Plot a comparison of true and predicted signals for the entire sequence
    plt.figure(figsize=(12, 6))
    plt.plot(y.flatten(), label="True Clean Signal", linestyle='--')
    plt.plot(predictions, label="Predicted Clean Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title("True vs. Predicted Clean Signal for the Entire Dataset")
    plt.legend()
    plt.show()

    # Save the DataFrame with predictions and accuracy to a new CSV file
    data.to_csv(output_path, index=False)
    print(f"Predicted signals with accuracy saved to {output_path}")

    return mse, data  # Return the MSE and updated DataFrame with predictions and accuracy

# Call the function with the specified paths and columns
mse, updated_data = predict_clean_signals(
    model_path=model_path, 
    dataset_path=dataset_path, 
    noisy_signal_column="Noisy Signal", 
    clean_signal_column="Clean Signal"
)

# Print the Mean Squared Error to see the performance
print(f"Mean Squared Error: {mse}")
