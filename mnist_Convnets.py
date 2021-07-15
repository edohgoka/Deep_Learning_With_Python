# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 21:49:50 2021

@author: goka
"""

"""

The book DEEP LEARNING with PYTHON

"""

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# Instantiating a small convnets
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1))) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

print(model.summary())


# Adding a classifier on top of the convnets
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation = 'softmax'))

# Show the architecture of the network 
print(model.summary())

# Training the convnets on the MNIST images
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_images[0].shape)
print(train_images[0][0].shape)
print(type(train_images[0][0][0]))


train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy']
              )

model.fit(train_images, train_labels, epochs = 5, batch_size= 64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
print(test_loss)