# Keras code for MNIST digit classification using CNNs
import numpy as np 
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Compute the number of labels
num_labels = len(np.unique(y_train))

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Get image dimensions
image_size = x_train.shape[1]

# Resize input and normalize image values to unit interval
# Input to CNN is of shape (instances, h, w, c)
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
# Image is processed as is (square grayscale)
input_shape = (image_size, image_size, 1)   # 1 channel (c) - BW image
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64               # number of feature maps
dropout = 0.2              # dropout rate

# Model is a stack of CNN-ReLU-MaxPooling
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
# Dropout added as a regularizer
model.add(Dropout(rate=dropout))
# Output layer is a 10-dimensional one-hot vector
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()
#plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# Loss function for one-hot vector
# Use of ADAM optimizer
# Accuracy is a good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the network
model.fit(x_train, y_train, batch_size=batch_size, epochs=10)

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\nTest accuracy: {0:.1f}%'.format(100.0 * acc))

