import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate_ssvep_model(model, test_data):
    """
    Evaluates the given model on the provided test data.

    Parameters:
    - model: Trained TensorFlow model to predict SSVEP signals.
    - test_data: DataFrame containing the test data with the necessary features and target columns.

    Returns:
    - mse: Mean Squared Error between the predicted and true clean signals.
    """
    
    # Select relevant columns for features and target
    features = test_data[[
        "Clean Signal with Harmonics",
        "First Harmonic Amplitude",
        "Second Harmonic Amplitude",
        "Phase of 1st Harmonic",
        "Phase of 2nd Harmonic"
    ]].values
    target = test_data["Clean Signal"].values

    # Reshape data for testing
    timesteps = 1000
    feature_count = features.shape[1]

    X = features.reshape(-1, timesteps, feature_count)
    y = target.reshape(-1, timesteps)

    # Use the test split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error for evaluation
    mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    print(f"Test Mean Squared Error: {mse}")

    # Plot a comparison of true and predicted signals for a sample
    sample_index = 0  # choose a sample to visualize
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[sample_index], label="True Clean Signal", linestyle='--')
    plt.plot(predictions[sample_index], label="Predicted Clean Signal")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.title("True vs. Predicted Clean Signal for Sample")
    plt.legend()
    plt.show()

    return mse

def main(model_path, dataset_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load the test data
    test_data = pd.read_parquet(dataset_path)

    # Evaluate the model with the test data
    mse = evaluate_ssvep_model(model, test_data)
    print(f"Mean Squared Error on Test Data: {mse}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test a model with specified test data.")
    parser.add_argument("model_path", type=str, help="Path to the model to be tested")
    parser.add_argument("dataset_path", type=str, help="Path to the test data file (parquet format)")
    args = parser.parse_args()

    # Run main with arguments
    main(args.model_path, args.dataset_path)
