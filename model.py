import tensorflow as tf
from tensorflow.keras import Model
import nodepy.linear_multistep_method as lm
import numpy as np
import timeit

tf.keras.backend.set_floatx('float32')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # for MAC OS

class lmmNet:
    """
    A feed-forward Neural Network with one hidden layer embedded in the framework of linear multistep method.
    This model learns full dynamics of a dynamical system and can predict the derivatives at any point.
    """
    
    def __init__(self, h, X, M, scheme, hidden_units):
        """
        Args:
        h -- step size
        X -- data array with shape S x N x D 
        M -- number of LMM steps
        scheme -- the LMM scheme (either AB, AM, or BDF)
        hidden_units -- number of units for the hidden layer
        
        """
        self.h = h
        self.X = X
        self.M = M # number of time steps
        
        # get the number of trajectories, discrete time instances, and number of feature dimensions
        self.S = X.shape[0]
        self.N = X.shape[1]
        self.D = X.shape[2]
        
        # load LMM coefficients from NodePy
        # https://nodepy.readthedocs.io/en/latest/
        if scheme == 'AB':
            coefs = lm.Adams_Bashforth(M)
        elif scheme == 'AM':
            coefs = lm.Adams_Moulton(M)
        elif scheme == 'BDF':
            coefs = lm.backward_difference_formula(M)
        else:
            raise Exception('Please choose a valid LMM scheme')
        
        self.alpha = np.float32(-coefs.alpha[::-1])
        self.beta = np.float32(coefs.beta[::-1])
        
        class DenseModel(Model):
            """
            A simple feed-forward network with 1 hidden layer
            
            Arch:
            * 256 hidden units
            * input units and output units correspond to the dimensionality
            """
            def __init__(self, D):
                super(DenseModel, self).__init__()
                self.D = D

                self.d1 = tf.keras.layers.Dense(units=hidden_units, activation=tf.nn.tanh, input_shape=(self.D,))
                self.d2 = tf.keras.layers.Dense(units=self.D, activation=None)

            def call(self, X1):
                A = self.d1(X1)
                A = self.d2(A)
                return A
        
        self.nn = DenseModel(self.D)
                
        self.opt = tf.keras.optimizers.Adam()
        
    def get_F(self, X):
        """
        Output of the NN/ML model.
        
        Args:
        - X: the data matrix with shape S x (N-M) x D
        
        Output:
        - F: the output dynamics with shape S x (N-M) x D
        """

        assert X.shape == (self.S, self.N - self.M, self.D)
        
        X1 = tf.reshape(X, [-1, self.D])
        F1 = self.nn(X1)
        
        return tf.reshape(F1, [self.S, -1, self.D])
    
    def get_Y(self, X):
        """
        The linear difference (residual) operator.
        
        Args:
        - X: the data matrix with shape S x N x D
        """
        
        M = self.M
        
        # compute the difference operator
        # broadcasting from M to N to get an array for all n
        # Y has shape S x (N - M) x D
        Y = self.alpha[0] * X[:, M: ,:] + self.h * self.beta[0] * self.get_F(X[:, M:, :]) # for m = 0
        
        # sum over m from m = 1
        for m in range(1, M+1):
            Y += self.alpha[m] * X[:, M-m:-m, :] + self.h * self.beta[m] * self.get_F(X[:, M-m:-m, :])
        
        return self.D * tf.reduce_mean(tf.square(Y))
    
    def train(self, epochs):
        """
        Fit the model PyTorch-style
        """
        
        start_time = timeit.default_timer()
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                self.loss = self.get_Y(self.X)
            grads = tape.gradient(self.loss, self.nn.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.nn.trainable_variables))
            
            if epoch % 100 == 0:
                elapsed_time = timeit.default_timer() - start_time
                #print('Epoch: %d, Time: %.2f, Loss: %.4e' %(epoch, elapsed_time, self.loss))
                #tf.print(self.loss)

        
    def predict(self, X_reshaped):
        """
        Args:
        - X_reshaped with shape S(N-M+1) x D
        """
        return self.nn(X_reshaped)