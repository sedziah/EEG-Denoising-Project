# network_structure.py

import tensorflow as tf
from tensorflow.keras import layers

def RNN_lstm(timesteps, feature_count):
    model = tf.keras.Sequential([
        layers.Input(shape=(timesteps, feature_count)),
        layers.LSTM(64, return_sequences=False),  # Set return_sequences=False to get a single output per sequence
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')  # Output a single value
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.9), loss='mean_squared_error')
    return model

