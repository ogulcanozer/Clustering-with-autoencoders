import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np


class make_ae:

    def Encoder(self, f_dim, h_dims, activs):
        
        encode = Dense (h_dims[0], activation = activs[0])(f_dim)
        if(len(h_dims)>1):
            for i in range(1, len(h_dims)):
                encode = Dense (h_dims[i], activation = activs[i])(encode)
        return Model(f_dim, encode)
        
    def Decoder(self, o_dim, h_dims, activs):

        decode = Dense (h_dims[0], activation = activs[0])(o_dim)
        if(len(h_dims)>1):
            for i in range(1, len(h_dims)):
                decode = Dense (h_dims[i],activation = activs[i])(decode)
        return Model(o_dim, decode)

    def __init__(self, f_dim, e_dims, o_dim ,d_dims, e_acts, d_acts):
        input_h = Input(shape=(f_dim,))
        output_h = Input(shape=(o_dim,))
        self.encoder = self.Encoder(input_h,e_dims,e_acts)
        self.decoder = self.Decoder(output_h,d_dims,d_acts)       
        self.autoencoder = Model(input_h, self.decoder(self.encoder(input_h)))
        print(self.autoencoder.summary())
