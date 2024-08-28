pip install adversarial-robustness-toolbox
tf.compat.v1.disable_eager_execution()
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import KerasClassifier

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Preprocess the data
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

  # pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Select an image from the test set
image_index = 3460
original_image = x_test[image_index]
original_image = original_image.reshape(1, 32, 32, 3).astype('float32') / 255.0
original_label = y_test[image_index]

# Create a KerasClassifier for the model
classifier = KerasClassifier(model=model, clip_values=(0, 1))

start_time = datetime.datetime.now()

# Attack parameters
attack = CarliniL2Method(classifier=classifier, targeted=True, confidence=0.0, learning_rate=0.01, max_iter=1000, binary_search_steps=9, initial_const=1e-2)

# Create a target label (random target class)
num_classes = 10

target_label = np.zeros((1, num_classes))
target_class = np.random.randint(0, num_classes)
target_label[0, target_class] = 1

# Generate the adversarial example
adversarial_example = attack.generate(x=original_image, y=target_label)

# Predict the label for the adversarial example
adversarial_predictions = model.predict(adversarial_example)
adversarial_label = np.argmax(adversarial_predictions)

end_time = datetime.datetime.now()
Elapsed_time = end_time - start_time

# Print the results
print("Original Label:", np.argmax(original_label))
print("Adversarial Label:", adversarial_label)
print(f"Task took {Elapsed_time}.")

