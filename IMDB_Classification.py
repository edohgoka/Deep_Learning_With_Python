# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:31:07 2021

@author: goka
"""

"""

The book DEEP LEARNING with Python 

"""

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import  layers
from keras import optimizers
from keras import losses
from keras import metrics 
import matplotlib.pyplot as plt 

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


# Exploration of the data set
print(train_data[0])
print(train_data.shape)

print(train_labels)

print(test_data.shape)
print(test_labels.shape)

# Decode one of these reviews back in english
word_index = imdb.get_word_index() # word index is a dictionary mapping words to an integer index
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()] # Reverse it, mapping integer indices to words
    )

decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
    ) # Decodes the review. Note that the indices are offset by 3 because 0, 1, and 2 are reserved 
      # indices for 'padding', 'start of sequence', and 'unknown'
 

# Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension)) # Creates an all-zero matrix of shape 
                                                    # (len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # Sets specific indices of results[i] to 1s
        
    return results

x_train = vectorize_sequences(train_data) # Vectorizing training data
x_test = vectorize_sequences(test_data) # Vectorizing test data

print(x_train[0])

# Vectorizing the labels 
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# the model definition
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# Compiling the model
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# configuring the optimizer 
model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Using custom losses and metrics 
model.compile(optimizer = optimizers.RMSprop(lr=0.001),
              loss = losses.binary_crossentropy, 
              metrics = [metrics.binary_accuracy])

##### Validation your approach 

# Setting aside a validation set 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# Training the model 
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict)
print(history_dict.keys())

# Plotting the training and validation loss 
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy 
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# retraining a model from scratch 

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test)

print(results)

print(model.predict(x_test))
