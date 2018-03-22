from keras.optimizers import Adam, Nadam
from keras import backend as K
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.initializers import glorot_uniform, zero
import numpy as np
from keras.regularizers import l2
from IPython.core.debugger import set_trace
from keras.models import Model
from keras.layers import (Conv1D, LSTM, BatchNormalization, Flatten, Dense, Activation, 
                          Input, concatenate)
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Nadam
from clr_callback import CyclicLR
import keras.backend as K
import tensorflow as tf

def conv_layer(prev):
    x = Conv1D(32, 1, kernel_regularizer=l2(10e-4),
               bias_regularizer=l2(10e-4))(prev)
    x = BatchNormalization(x)
    x = concatenate([x, prev], axis=-1)
    x = Activation("relu")(x)
    return x


def res_layer(prev):
    x = Conv1D(32, 1, kernel_regularizer=l2(10e-4),
               bias_regularizer=l2(10e-4))(prev)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(32, 1, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4))(x)
    x = BatchNormalization()(x)
    x = concatenate([x, prev], axis=-1)
    x = Activation("relu")(x)
    return x


def lstm_layer(prev):
    x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4)))(prev)
    x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4)))(x)
    return x


def lstm_value_head(prev):
    x = Bidirectional(LSTM(1, return_sequences=False, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4)))(prev)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(1, activation="tanh")(x)
    return x


def lstm_policy_head(prev, output_length):
    x = Bidirectional(LSTM(1, return_sequences=True, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4)))(prev)
    x = Bidirectional(LSTM(1, return_sequences=False, kernel_regularizer=l2(
        10e-4), bias_regularizer=l2(10e-4)))(x)
    x = Dense(output_length, activation="softmax",
              kernel_regularizer=l2(10e-4), bias_regularizer=l2(10e-4))(x)
    return x

def create_net(time_steps, input_length, output_length, num_layers=8):
    inp = Input(shape=(time_steps, input_length))
    x = inp
    for _ in range(num_layers):
        x = lstm_layer(x)
    policy = lstm_policy_head(x, output_length)
    value = lstm_value_head(x)
    
    model = Model(inputs=inp, outputs=[policy, value])
    
    model.compile(optimizer=Nadam(), loss = ["categorical_crossentropy", "mse"], loss_weights = [.5, .5], 
                  metrics=None)
    
    return model