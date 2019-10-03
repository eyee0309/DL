# Denoising autoencoder implementation in Keras

# Import required libraries for construction of model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K

np.random.seed(1337)

# Load the MNIST dataset
# NB: Shape of x_train is (instances, h, w)
(x_train, _), (x_test, _) = mnist.load_data()

# Reshape to (28, 28, 1) and normalize input images
image_size = x_train.shape[1]
# Shape of x_train required for CNN model is (instances, h, w, c)
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal distribution
# centered at 0.5 and std = 0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

# Adding noise may exceed normalized pis=xel values > 1.0 or < 0.0
# Clip pixels values > 1.0 to 1.0 and < 0.0 to 0.0
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# Network parameters
input_shape = (image_size, image_size, 1)    # (h, w, c)
batch_size = 32
kernel_size = 3
latent_dim = 16
# Encoder/decoder number of filters per CNN layer
layer_filters = [32, 64]

# Build the autoencoder model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Encoder design
#==========================================================================
# First build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# Shape info needed to build decoder model
# so we don't have to do hand calculation
# The input to the decoder's first Conv2DTranspose layer
# will have this shape
# Shape is (7,7,64) which is processed by
# the decoder back to (28,28,1)
shape = K.int_shape(x)

# Generate latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate encoder model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Decoder design
#=========================================================================
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
# Use the shape (7,7,64) that was saved earlier
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
# From vector to suitable shape for transposed conv
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    
# Reconstruct the input
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# Instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Construct autoencoder
#========================================================================
# autoencoder = encoder + decoder
# Instantiate autoencoder model
autoencoder = Model(inputs,
                    decoder(encoder(inputs)),
                    name='autoencoder')

autoencoder.summary()

# Mean square error (MSE) loss function, ADAM optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=10,
                batch_size=batch_size)

# Predict the autoencoder output from the corrupted test images
x_decoded = autoencoder.predict(x_test_noisy)

# Visualization of the results
# Get and process some examples of predictions
# 3 sets of images with 9 MNIST digits
# 1st rows - original images
# 2nd rows - images corrupted by noise
# 3rd rows - denoised images
rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num],
                       x_test_noisy[:num],
                       x_decoded[:num]])
imgs = imgs.reshape((rows*3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows*3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs*255).astype(np.uint8)

# Plot
plt.figure()
plt.title('Original images: top rows, '
          'Corrupted input: middle rows, '
          'Denoised input: third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()