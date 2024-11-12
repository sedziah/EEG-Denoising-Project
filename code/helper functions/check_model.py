import tensorflow as tf

model_path = "ssvep_denoising_model_with_all_features.keras"

# Load the model
model = tf.keras.models.load_model(model_path)

# Print the model summary
model.summary()
