import keras

from keras import backend as K
from keras.layers import Layer, InputLayer, Reshape
from keras.models import Sequential
import numpy as np
import scipy.linalg

import tensorflow as tf

def weightedIterativeStatistic(point_list, weights, \
        iterative_stat_function, xlen=128):
    mean = iterative_stat_function(tf.reshape(point_list[0], (1,-1)), \
            tf.reshape(point_list[1], (1,-1)), \
            weights[0])
    for i in range(2, xlen):
        mean = iterative_stat_function(tf.reshape(mean, (1,-1)),\
                tf.reshape(point_list[i], (1,-1)), \
                weights[i])
    return mean

def grassmanGeodesic(X, Y, t):
    '''
    X, Y are numpy arrays or tf tensors of the same shape. 
    t is a None x 1 array/tensor.
    '''
    mult        = tf.matmul(tf.transpose(X), Y)
    invs        = tf.linalg.inv(mult)
    svd_t       = tf.matmul(Y, invs) - X
    U,s,V       = tf.svd(svd_t)
    theta       = tf.atan(s[0])
    sin_term    = tf.matmul(tf.reshape(U, (1,-1)), \
                            tf.diag(tf.sin(theta*t)))
    xv          = tf.matmul(X, V)
    costheta    = tf.cos(theta*t, dtype=np.float32)
    costheta    = tf.reshape(costheta, (-1,1))
    qr_term     = tf.matmul(xv, costheta) + sin_term
    return qr_term

def stiefelGeodesicApprox(X, Y, t):
    X               = tf.cast(X, dtype=tf.float32)
    X               = tf.reshape(X, (1,-1))
    Y               = tf.cast(Y, dtype=tf.float32)
    Y               = tf.reshape(Y, (1,-1))
    t               = tf.cast(t, dtype=tf.float32)
    t               = tf.reshape(t, (1,-1))
    dualtranspose   = tf.matmul(tf.transpose(Y), X) + \
                      tf.matmul(tf.transpose(X), Y)
    lift            = Y - 0.5*tf.matmul(X, dualtranspose)
    scale           = t*lift
    Ishape          = scale.shape.as_list()[0]    
    scale_term      = tf.eye(int(Ishape)) + \
                      tf.matmul(tf.transpose(scale), scale)
    retract         = tf.matmul((X+scale), tf.linalg.inv(scale_term))
    return retract

def weightedFrechetMeanUpdate(previous_mean, new_point, weight, \
                            geodesic_generator=stiefelGeodesicApprox):
    return geodesic_generator(previous_mean, new_point, weight)

####### Custom Convolutional Layer #######
class wFMConv(Layer):
    def __init__(self, output_dim, in_shape=(-1,128,2), **kwargs):
        self.output_dim = output_dim
        self.in_shape= in_shape
        super(wFMConv, self).__init__(**kwargs)

    # input_shape = (None, 128, 2)
    def build(self, input_shape):
        # Trainable weight variable for this layer
        self.kernel = self.add_weight(\
                            name='kernel', \
                            shape=(input_shape[1], 1), \
                            initializer=tf.random_uniform_initializer(\
                                            dtype=tf.float32), \
                            trainable=True,
                            dtype=np.float32)
        super(wFMConv, self).build(input_shape)

    #NOTE: Change xlen to x.shape[0]
    def call(self, x):
        out         = weightedIterativeStatistic(x, self.kernel, \
                            weightedFrechetMeanUpdate, xlen=128)
        out         = tf.reshape(out, (-1,out.shape[0],out.shape[1]))
        return out

    # output_dim = 128
    # output_shape = (None, 128, 2)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
############################################


####### Custom Invariant Layer #############
#def invariantLayer(Layer):
#    def __init__(self)
#        pass
#
#############################################l


def build_model(in_shape, conv_input_dim=2, conv_output_dim=2, \
        optimizer='adam', metrics=['acc'], loss='mse'):
    
    in_layer = Input(in_shape)
    act1 = Activation()(in_layer)
    conv1 = wFMConv(2)(act1)
    reshaped1 = Reshape((-1,128,2))(conv1)
    flat1 = Flatten()(reshaped1)
    dense1 = Dense(11)(flat1) 
    softout = Activation('softmax')(dense1)
    sigout = Activation('sigmoid')(dense1)
    swishout = Multiply()([dense1, sigout]) 

    model = Model([in_layer], [softout, swishout])
    model.compile(optimizer=optimizer, metrics=metrics, loss=loss)
    model.summary()

    return model

