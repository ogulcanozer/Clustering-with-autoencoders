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
from ae import make_ae
from data import data_struct as DS
##************************

#Read and seperate the mnist dataset.
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)
ae = make_ae(784,[32],32,[784],['relu'],['sigmoid'])
print(ae.autoencoder.summary())
ae.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
ae.autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = ae.encoder.predict(x_test)
decoded_imgs = ae.decoder.predict(encoded_imgs)
print (encoded_imgs.shape)
print (decoded_imgs.shape)
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