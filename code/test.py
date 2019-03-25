

##************************
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets
import sklearn
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from data import data_struct as DS
from encoder import make_encoder as end
from decoder import make_decoder as ded
##************************

#Read and seperate the mnist dataset.
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
input_h = Input(shape=(784,))
enc = end(input_h,[128,64,16],["relu","relu","relu"])
dec = ded(enc.encode,[64,128,784],["relu","relu","relu"])
print(enc.encode)
print(dec.decode)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)
autoencoder = Model (input_h, dec.decode)
encoder = Model(input_h, enc.encode)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(16,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-3]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
print(autoencoder.summary())
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()