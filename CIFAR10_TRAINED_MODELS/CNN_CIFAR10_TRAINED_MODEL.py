#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


# In[15]:


# Load the Cifar10 dataset
(x_train, y_train), (x_test, y_test) =datasets.cifar10.load_data()


# Preprocessing the Cifar10 dataset
x_train = x_train.reshape(-1, 32, 32, 3).astype("float32")/255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype("float32")/255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# In[16]:


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=256, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[17]:


model.summary()


# In[18]:


# Plot the model architecture to a file
from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[19]:


from datetime import datetime
Start_time_1 = datetime.now()

model.compile(optimizer =Adam(learning_rate=0.001),
             loss = CategoricalCrossentropy(),
             metrics = ["accuracy"])


# In[20]:


model.fit(x_train, y_train, batch_size=512, epochs=20, validation_split=0.1)


# In[21]:


End_time_1 = datetime.now()

print('Time taken to train the model: {}'.format(End_time_1 - Start_time_1))

test_acc = model.evaluate(x_test, y_test)

print("Loss accuracy, Test accuracy: \n", test_acc)


# In[22]:


model.save("cifar10_trained_runtimeT4GPU_model.h5")

