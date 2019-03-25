import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np

class make_decoder:

    def __init__(self, encode, h_dims, activs):
        self.decode = Dense (h_dims[0], activation = activs[0])(encode)
        print('curr decode : 0',)
        print(self.decode)
        for i in range(1, len(h_dims)):
            self.decode = Dense (h_dims[i],activation = activs[i])(self.decode)
            print('curr decode :', i)
            print(self.decode)