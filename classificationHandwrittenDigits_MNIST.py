# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:08:07 2021

@author: goka
"""


"""

The book Deep Learning with Python 

"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Loading the MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Exploration of the data
print(train_images.shape)
print(len(train_images))

print(train_labels)

print(test_images.shape)
print(len(test_images))

print(test_labels)

# Displaying the fourth digit
plt.imshow(train_images[0], cmap = plt.cm.binary)
plt.show()

# The network achitecture 
network = models.Sequential()
network.add(layers.Dense(512, activation = "relu", input_shape=(28*28,)))
network.add(layers.Dense(10, activation = "softmax"))

# The compilation step
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

# Preparing the image data 
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float') /255


# Preparing the labels 
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs = 5, batch_size=128)

# How the model performs on the test sets
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

