#_______________________________________________________________________________
# CE888 Project |        ae.py       | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________
import ae_config as ae_conf
import numpy as np
import tensorflow as tf

class ae:

    def __init__(self, dim_feat, dim_hidden):
        
        # Weights and biases for the hidden layer(h) and the output(y).
        self.h_W = tf.Variable(tf.random_normal([dim_feat, dim_hidden]))
        self.h_b = tf.Variable(tf.random_normal([dim_hidden]))
        self.y_W = tf.Variable(tf.random_normal([dim_hidden, dim_feat]))
        self.y_b = tf.Variable(tf.random_normal([dim_feat]))

        # Initialization of the input(x), hidden(h) and output(y) layers.
        self.x = tf.placeholder('float', [None, dim_feat])
        self.h = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.h_W), self.h_b))
        self.y = tf.matmul(self.h, self.y_W) + self.y_b
        self.d = tf.placeholder('float', [None, dim_feat])
        
        # Initialization of optimizer and loss function.
        self.mse = tf.reduce_mean(tf.square(self.y - self.d))
        self.opt = tf.train.AdagradOptimizer(ae_conf.ae_param.learning_rate).minimize(self.mse)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

    #Train the auto encoder.
    def train(self, X_train, x_test, batch_size):
        # Initialize parameters
        self.sess.run(self.init)
        
        # Repeat number of times declared in config.
        for epoch in range(ae_conf.ae_param.ae_epoch):
            e_loss = 0
            for i in range(int(X_train.shape[0]/batch_size)):
                epoch_input = X_train[ i * batch_size : (i + 1) * batch_size ]
                _, c = self.sess.run([self.opt, self.mse], feed_dict={self.x: epoch_input, self.d: epoch_input})
                epoch_loss += c
                print('Epoch', epoch, '/', ae_conf.ae_param.ae_epoch, 'loss:',e_loss)
        
        #************* USE TO SAVE AND DISPLAY THE MODEL *************#
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        writer.flush()

    # Encode a given image.
    def encode_image(self, image):
        encoded_image = self.sess.run(self.h, feed_dict={self.x:[image]})
        return encoded_image
    # Decode a given image.
    def decode_image(self, image):
        decoded_image = self.sess.run(self.y, feed_dict={self.x:[image]})
        return decoded_image


#-------------------------------------------------------------------------------
# End of ae.py 
#-------------------------------------------------------------------------------
#_______________________________________________________________________________
# ACKNOWLEDGEMENTS
#_______________________________________________________________________________
#
#%https://www.tensorflow.org/guide/low_level_intro
#
#%CE802-Machine Learning and Data Mining Lecture2_MLPs.pdf
#_______________________________________________________________________________