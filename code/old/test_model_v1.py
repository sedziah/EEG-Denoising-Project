import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def evaluate_ssvep_model(model, test_data):
    # Select relevant columns for features and target
    features = test_data[[
        "Noisy Signal",
        "Harmonic 1",
        "Harmonic 2"
    ]].values
    target = test_data["Clean Signal"].values

    # Set timesteps based on data availability
    timesteps = 100  # Adjust this based on your data size
    feature_count = features.shape[1]

    # Calculate the number of complete samples we can form
    num_samples = features.shape[0] // timesteps
    X = features[:num_samples * timesteps].reshape(num_samples, timesteps, feature_count)
    y = target[:num_samples * timesteps].reshape(num_samples, timesteps)

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

    # Load the test data, handling either Parquet or CSV format
    file_extension = os.path.splitext(dataset_path)[1].lower()
    if file_extension == ".parquet":
        test_data = pd.read_parquet(dataset_path)
    elif file_extension == ".csv":
        test_data = pd.read_csv(dataset_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    # Evaluate the model with the test data
    mse = evaluate_ssvep_model(model, test_data)
    print(f"Mean Squared Error on Test Data: {mse}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test a model with specified test data.")
    parser.add_argument("model_path", type=str, help="Path to the model to be tested")
    parser.add_argument("dataset_path", type=str, help="Path to the test data file (CSV or Parquet format)")
    args = parser.parse_args()

    # Run main with arguments
    main(args.model_path, args.dataset_path)
