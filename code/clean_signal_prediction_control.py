import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Define paths to model and dataset
model_path = "ssvep_denoising_model_with_all_features.keras"
dataset_path = "control_data.csv"
output_path = "control_data_predicted.csv"  # Define output path for the new CSV file

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

    # Reshape for model input format (batch_size, timesteps, features)
    X = model_input.reshape(1, -1, 5)  # Shape to (1, total_timesteps, 5)
    y = clean_signal.reshape(1, -1)    # Shape to (1, total_timesteps)

    # Generate predictions on the entire dataset using direct model call for single input
    predictions = model(X, training=False).numpy().flatten()

    # Calculate Mean Squared Error for evaluation
    mse = mean_squared_error(y.flatten(), predictions)
    print(f"Test Mean Squared Error: {mse}")

    # Add predictions as a new column in the original DataFrame
    data['Predicted Clean Signal'] = predictions

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

# Save the DataFrame with predictions and accuracy to a new CSV file
updated_data.to_csv(output_path, index=False)
print(f"Predicted signals with accuracy saved to {output_path}")