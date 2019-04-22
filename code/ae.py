import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input,Dropout
from keras import regularizers
import numpy as np


class make_ae:

    def Encoder(self, f_dim, h_dims, activs):
        
        self.encode = Dense (h_dims[0],activation = activs[0])(f_dim)
        if(len(h_dims)>1):
            for i in range(1, len(h_dims)):
                self.encode = Dense (h_dims[i],activation = activs[i])(self.encode)
        return self.encode
        
    def Decoder(self, o_dim, h_dims, activs):

        self.decode = Dense (h_dims[0], activation = activs[0])(o_dim)
        if(len(h_dims)>1):
            for i in range(1, len(h_dims)):
                self.decode = Dense (h_dims[i],activation = activs[i])(self.decode)
        return self.decode

    def __init__(self, f_dim, e_dims, o_dim ,d_dims, e_acts, d_acts):
        input_e = Input(shape=(f_dim,), dtype='float32', name='main_input')
        
        self.encode = self.Encoder(input_e,e_dims,e_acts)
        input_d = self.encode
        self.decode = self.Decoder(input_d,d_dims,d_acts)
        self.encoder = Model(input=input_e, output=[self.encode])
        self.autoencoder = Model(inputs=input_e,outputs=[self.decode])
        print(self.autoencoder.summary())
