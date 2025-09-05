import tensorflow as tf

# Check if TensorFlow detects any GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
else:
    print("TensorFlow did not detect any GPUs.")

# Enable device placement logging to confirm operations run on the GPU
tf.debugging.set_log_device_placement(True)

# Define a simple Keras model and train it
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(10,))
])
model.compile(optimizer='adam', loss='mse')
model.fit(tf.random.normal((32, 10)), tf.random.normal((32, 10)), epochs=1)