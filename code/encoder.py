import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np

class make_encoder:

    def __init__(self, f_dim, h_dims, activs):    
        self.encode = Dense (h_dims[0], activation = activs[0])(f_dim) 
        print('curr encode : 0',)
        print(self.encode)
        for i in range(1, len(h_dims)):
            self.encode = Dense (h_dims[i], activation = activs[i])(self.encode)
            print('curr encode :', i)
            print(self.encode)
