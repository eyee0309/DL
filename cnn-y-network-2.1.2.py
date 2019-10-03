# Y-network implementation using the Functional API for MNIST digit classification

# Use either CPU or GPU for computations in Keras
import tensorflow as tf
from keras import backend as K 

GPU = False               # boolean to use GPU
CPU = True                # boolean to use CPU

num_cores = 4

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU' : num_CPU, 'GPU' : num_GPU})

session = tf.Session(config=config)
K.set_session(session)

# Import libraries for modeling
import numpy as np 

from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Compute the number of labels
num_labels = len(np.unique(y_train))

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape and normalize input images
image_size = x_train.shape[1]                                      # shape of x_train is (instances, h, w)
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])     # input shape is (instances, h, w, c)
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
input_shape = (image_size, image_size, 1)    # each input is (h, w, c)
batch_size = 32
kernel_size = 3
dropout = 0.4         # dropout rate
n_filters = 32        # base number of feature maps

# Left branch of Y network
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 3 layers of Conv2D-Droput-MaxPooling2D
# Number of filters doubles after each layer (32-64-128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',                   # input and output images are same size
               activation='relu')(x)
    x = Dropout(rate=dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# Right branch of Y network
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# 3 layers of Conv2D-Droput-MaxPooling2D
# Number of filters doubles after each layer (32-64-128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)
    y = Dropout(rate=dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# Merge the left and right branches output of Y network
y = concatenate([x, y])
# Feature maps to vector before connecting to Dense layer
y = Flatten()(y)
y = Dropout(rate=dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# Build the model in functional API
model = Model([left_inputs, right_inputs], outputs)
# Verify the model using layer text description
model.summary()
# Verify the model using graph
# plot_model(model, to_file='cnn-y-network.png', show_shapes=True)

# Classifier loss, ADAM optimizer, classifier accuracy for metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model with input images and labels
model.fit([x_train, x_train],
           y_train,
           validation_data=([x_test, x_test], y_test),
           epochs=20,
           batch_size=batch_size)

# Model accuracy on test dataset
score = model.evaluate([x_test, x_test], y_test, batch_size=batch_size)
print('\nTest accuracy: {0:.1f}%'.format(100.0 * score[1]))   # score[0] = loss; score[1] = accuracy


    


