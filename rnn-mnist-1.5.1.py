# Keras code for MNIST digit classification using RNNs
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Compute the number of labels
num_labels = len(np.unique(y_train))

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Resize and normalize
# Input to RNN: (instances, time_steps, input_dim)
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size])
x_test = np.reshape(x_test, [-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
input_shape = (image_size, image_size)     # (time_steps, input_dim)
batch_size = 128
units = 256
dropout = 0.2         # dropout rate

# Model is RNN with 256 units, input is 28-dim vector 28 timesteps
model = Sequential()
model.add(SimpleRNN(units=units,
                    dropout=dropout,
                    input_shape=input_shape))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()
#plot_model(model, to_file='rnn-mnist.png', show_shapes=True)

# Loss function for one-hot vector
# Use SGD optimizer
# Accuracy is a good metric for classication tasks
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the network
model.fit(x_train, y_train, batch_size=batch_size, epochs=20)

# Evaluate the model
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\nTest accuracy: {0:.1f}%'.format(100.0 * acc))
