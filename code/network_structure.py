import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input

def RNN_lstm(datanum, feature_count):
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, feature_count)))  # Input shape is (timesteps, features)
    model.add(layers.LSTM(64, return_sequences=True))  # LSTM with 64 units

    model.add(layers.Flatten())
    
    model.add(layers.Dense(64))  # Dense layer with 64 neurons
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32))  # Additional dense layer with 32 neurons
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(datanum))  # Output layer to match output shape
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile with Adam optimizer and MSE loss

    model.summary()
    return model
