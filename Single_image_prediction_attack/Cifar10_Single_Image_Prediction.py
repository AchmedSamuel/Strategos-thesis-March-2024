import numpy as np
import tensorflow as tf
import csv
import pandas as pd
from tensorflow.keras import Sequential, datasets

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Preprocessing
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Preprocess the data for a single image
Image_index = 3460
original_image = x_test[Image_index]
original_image = original_image.reshape(1, 32, 32, 3).astype('float32') / 255.0
true_label_output = np.argmax(y_test[Image_index])

# Make predictions on the sample image
predictions = model.predict(original_image)
predicted_label = np.argmax(predictions)

# Displaying the results
print("Original Image:", y_test[Image_index])
print("True Label Output:", true_label_output)
print("Predicted Image:", predicted_label)
