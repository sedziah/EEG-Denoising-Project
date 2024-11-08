import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def predict_clean_signals(
    model_path, 
    dataset_path, 
    noisy_signal_column="Clean Signal with Harmonics", 
    clean_signal_column="Clean Signal"
):
    """
    Evaluates a trained model on the entire dataset as a single sequence.

    Parameters:
    - model_path (str): Path to the trained model (.keras file).
    - dataset_path (str): Path to the dataset file (.parquet or .csv).
    - noisy_signal_column (str): Column name for the noisy signal (input to the model).
    - clean_signal_column (str): Column name for the clean signal (ground truth).

    Returns:
    - mse (float): Mean Squared Error between predicted and actual clean signal.
    - predictions (numpy array): Array of predicted clean signal values.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load the dataset
    data = pd.read_parquet(dataset_path)

    # Extract the noisy signal as the input feature and the clean signal as the target
    noisy_signal = data[[noisy_signal_column]].values  # Extract as (total_timesteps, 1)
    clean_signal = data[clean_signal_column].values    # Extract as (total_timesteps,)

    # Reshape for model input format
    X = noisy_signal.reshape(1, -1, 1)  # Shape to (1, total_timesteps, 1)
    y = clean_signal.reshape(1, -1)     # Shape to (1, total_timesteps)

    # Generate predictions on the entire dataset
    predictions = model.predict(X).flatten()

    # Calculate Mean Squared Error for evaluation
    mse = mean_squared_error(y.flatten(), predictions)
    print(f"Test Mean Squared Error: {mse}")

    # Add predictions as a new column in the original DataFrame
    data['Predicted Clean Signal'] = predictions

    # Plot a comparison of true and predicted signals for the entire sequence
    plt.figure(figsize=(12, 6))
    plt.plot(y.flatten(), label="True Clean Signal", linestyle='--')
    plt.plot(predictions, label="Predicted Clean Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title("True vs. Predicted Clean Signal for the Entire Dataset")
    plt.legend()
    plt.show()

    return mse, data  # Return the updated DataFrame with predictions
