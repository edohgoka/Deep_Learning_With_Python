# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:06:11 2021

@author: goka
"""


# Instantinating the VGG16 convotional base

from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers 
from keras import optimizers
import matplotlib.pyplot as plt


conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))

print(conv_base.summary())

# Extracting features using the pretrained convotional base
base_dir = 'SmallerDatasets' # Directory where you'll 
                                                                  # store your smaller dataset

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale = 1/255.0)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1)* batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break # Note that because generators yield data indefinitely in a loop, you must break
                  # every image has been seen once
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

##############################################################################
## Defining and training the densely connected classifier
##############################################################################
model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_dim = 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr = 2e-05),
              loss='binary_crossentropy',
              metrics = ['acc'])

history = model.fit(train_features, train_labels,
                    epochs = 30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

##############################################################################
## Save the model
##############################################################################
# Saving the model 
model.save('cats_and_dogs_small_Pretrained_VGG16.h5')

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

##############################################################################
### Feature extraction with data augmentation
##############################################################################
# Instantinating the VGG16 convotional base

from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers 
from keras import optimizers
import matplotlib.pyplot as plt


conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))

print(conv_base.summary())

# Extracting features using the pretrained convotional base
base_dir = 'D:/DataScience/Pretrained_VGG16/SmallerDatasets' # Directory where you'll 
                                                                  # store your smaller dataset

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

print(model.summary())



conv_base.trainable = False

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255) # Note that the validation data shouldn't be augmented !

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (150,150),
    batch_size = 20,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary')

model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=2e-05),
              metrics = ['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs= 30,
    validation_data= validation_generator,
    validation_steps=50)

##############################################################################
## Save the model
##############################################################################
# Saving the model 
model.save('cats_and_dogs_small_Pretrained_VGG16_DataAugmentation.h5')

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

##############################################################################
##### Fine-tuning
##############################################################################
"""
Fine-tuning consists of unfreezing a few of the top layers of a frozen model
base used for feature extraction, and jointly training both the newly added part
of the model and these top layers.
"""

# Instantinating the VGG16 convotional base

from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers 
from keras import optimizers
import matplotlib.pyplot as plt


conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (150, 150, 3))

print(conv_base.summary())

## Freezing all layers up to a specific one
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
# Extracting features using the pretrained convotional base
base_dir = 'SmallerDatasets' # Directory where you'll 
                                                                  # store your smaller dataset

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255) # Note that the validation data 
                                                  # shouldn't be augmented !

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (150,150),
    batch_size = 20,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary')

## Fine-tuning the model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


model.compile(
    loss = 'binary_crossentropy',
    optimizer = optimizers.RMSprop(lr = 1e-05),
    metrics = ['acc'])

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs = 100,
    validation_data=validation_generator,
    validation_steps=50)


##############################################################################
## Save the model
##############################################################################
# Saving the model 
model.save('cats_and_dogs_small_Pretrained_VGG16_DataAugmentation_FineTuning.h5')

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
## These curves look noisy so we need to make them smooth
#########################################################

# Smooting the plot 
def smooth_curve(points, factor = 0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label = 'Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label = 'Smoothed Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label = 'Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label = 'Smoothed validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

######################################
## We can now evaluate this model
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc', test_acc)