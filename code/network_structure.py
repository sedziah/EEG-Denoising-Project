import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input

def RNN_lstm(datanum):
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, 1)))  # Input shape is (timesteps, features)
    model.add(layers.LSTM(64, return_sequences=True))  # Increased LSTM units for better capacity

    model.add(layers.Flatten())
    
    model.add(layers.Dense(64))  # Added more neurons for the dense layer
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32))  # Added more layers for complexity
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(datanum))  # Output layer should match the output shape
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile with Adam optimizer and MSE loss

    model.summary()
    return model
