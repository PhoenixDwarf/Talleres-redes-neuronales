import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([[0], [1]], dtype=float)
y = np.array([[0], [1]], dtype=float)

model = Sequential([
    Dense(units=1, input_shape=[1], activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=1000, verbose=0)

print('\nPredicciones:')
print(model.predict(x))
