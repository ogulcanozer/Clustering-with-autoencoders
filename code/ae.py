import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np

class ae:

    def __init__(self, f_dim, e_dims, d_dims, activs):
        input_h = Input(shape=(f_dim,))
        enc = end(input_h,e_dims,activs)
        dec = ded(enc.encode,d_dims,activs)       
        print(enc.encode)
        print(dec.decode)
        self.autoencoder = Model (input_h, dec.decode)
        self.encoder = Model(input_h, enc.encode)
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(e_dims[-1],))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-len(d_dims)]
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))