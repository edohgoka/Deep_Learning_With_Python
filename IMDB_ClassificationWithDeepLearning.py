# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:41:16 2021

@author: goka
"""


## Instantiating an Embedding layer
from keras.layers import Embedding 
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

embedding_layer = Embedding(1000, 64) # The embedding layer takes at least two arguments:
                                      # the number of possible tokens (here, 1000 : 1+maximum word index)
                                      # and the dimensionality of the embeddings (here, 64)

### Word index ------> Embedding layer -------> Corresponding word vector

# Loading the IMDB data for use with an Embedding layer 
max_features = 10000 # Number of words to consider ad features 
maxlen = 20 # Cuts off the text after this number of words (among the max features most common words)

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features) # Loads the data as lists of integers

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen) # Turns the lists of integers
                                                                         # into a 2D integer tensor of
                                                                         # shape (samples, maxlen)

# Using anEmbedding layer and classifier on the IMDB data
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) # Specifies the maximum input length to the
                                                    # Embedding layer so you can later flatten
                                                    # the embedding inputs. After the embedding
                                                    # layer, the activation have shape 
                                                    # (samples, maxlen, 8)

model.add(Flatten())  # Flattens the 3D tensor of embeddings into a 2D tensor of shape 
                      # (samples, maxlen*8)

model.add(Dense(1, activation = 'sigmoid')) # Adds the classifier on top 
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs = 10,
                    batch_size=32,
                    validation_split=0.2)


##############################################################################
## Save the model
##############################################################################
# Saving the model 
model.save('deepLearningIMDB_WordEmbedding.h5')

##############################################################################
## Plotting the results
##############################################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation_acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='validation_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#############################################################################