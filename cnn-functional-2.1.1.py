# Functional API in Keras used for implementation of CNN for MNIST digit classification

# Code to use CPU in Keras
use_GPU = True         # if false use CPU, else GPU
if not use_GPU:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import libraries need for building deep learning CNN classifier
import numpy as np 
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Compute the number of labels
num_labels = len(np.unique(y_train))

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape and normalize input images
image_size = x_train.shape[1]                # (instances, h, w)
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])  # (instances, h, w, c)
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
input_shape = (image_size, image_size, 1)    # (h, w, c)
batch_size = 128
kernel_size = 3
filters = 64              # number of feature maps
dropout = 0.3             # dropout rate

# Use functional API to build CNN layers
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
# Image to vector before connecting to dense layer
y = Flatten()(y)
# Dropout regularization
y = Dropout(rate=dropout)(y)
# Output layer involves softmax
outputs = Dense(num_labels, activation='softmax')(y)

# Buid the model by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)
# Network summary
model.summary()

# Classifier loss, ADAM optimizer, classifier accuracy for metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model with input images and labels
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=20,
          batch_size=batch_size)

# Model accuracy
score = model.evaluate(x_test, y_test, batch_size=batch_size)     # scorce[0] = loss; score[1] = accuracy
print('\nTest accuracy: {0:.1f}%'.format(100.0 * score[1]))
