# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:14:15 2021

@author: goka
"""

"""

The book DEEP LEARNING with Python  

"""

from keras.datasets import reuters
import numpy as np 
from keras.utils import to_categorical
from keras import layers
from keras import models
import matplotlib.pyplot as plt


# loading the REUTERS Dataset 
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Exploring the dataset 
print(train_data.shape)
print(len(train_data[0]))

print(test_data.shape)
print(len(test_data[0]))

print(train_data[0])
print(train_labels.shape)
print(type(train_labels[0]))
print(train_labels[0])
print(train_labels)


# Decoding the newswires back in text 
word_index = reuters.get_word_index()

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
    )

decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
    ) # Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for "padding"
      # "start of sequence", and "unknown"


## preparing the data

# Encoding the data 

def vectorize_sequences(sequences, dimension=10000):
    
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    
    return results

x_train = vectorize_sequences(train_data) # vectorized training data
x_test = vectorize_sequences(test_data) # vectorized test data


# Vectorize the labels 

def to_one_hot(labels, dimension=46):
    
    results = np.zeros((len(labels), dimension))
    
    for i, label in enumerate(labels):
        results[i, label] = 1.
    
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# we can also use the built-in way from keras
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)


###
### Model definition 
###
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Compiling the model 
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Validating the model

# Setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Training the model 
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val)
                    )

history_dict = history.history
# Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.figure(0)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
plt.figure(1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()

plt.show()

# Retraining the modl from scratch 

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
              )

model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# Generating predictions on new data

predictions = model.predict(x_test)

print(predictions.shape)
print(predictions[0].shape)

print(np.sum(predictions[0]))

print(np.argmax(predictions[0]))


# Different way to handle the labels and the loss
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model2 = models.Sequential()
model2.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
model2.add(layers.Dense(64, activation = 'relu'))
model2.add(layers.Dense(46, activation = 'softmax'))

model2.compile(optimizer = 'rmsprop',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

model2.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512
          )

results_2 = model2.evaluate(x_test, y_test)
print(results_2)
