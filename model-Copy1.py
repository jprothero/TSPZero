from keras.models import Model
from keras.layers import (Conv1D, LSTM, BatchNormalization, Flatten, Dense, Activation, 
                          Input, concatenate)
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Nadam
from clr_callback import CyclicLR
import keras.backend as K


# Idea: try a diff activation later. Relu is fine for now though
# need embedding layer to specify layer types



def conv_layer(prev):
    x = Conv1D(32, 1)(prev)
    x = BatchNormalization(x)
    x = concatenate([x, prev], axis=-1)
    x = Activation("relu")(x)
    return x

def res_layer(prev):
    x = Conv1D(32, 1)(prev)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(32, 1)(x)
    x = BatchNormalization()(x)
#     not sure how this will work, might need to be an add
# seems like sizes wont match so I'm a bit confused
    x = concatenate([x, prev], axis=-1)
    x = Activation("relu")(x)
    return x

def lstm_layer(prev):
    x = Bidirectional(LSTM(32, return_sequences=True))(prev)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    return x

def lstm_value_head(prev):
#     print("might need more lstm_units")
    x = Bidirectional(LSTM(1, return_sequences=False))(prev)
#     print(x.shape)
#     x = Flatten()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
#     in the image it says no batch norm, but it seems like that would be better...
# going to try adding it
    x = Dense(1, activation="tanh")(x)
    return x

def lstm_policy_head(prev, output_length):
#     print("might need to have more lstm_units")
    x = Bidirectional(LSTM(1, return_sequences=True))(prev)
    x = Bidirectional(LSTM(1, return_sequences=False))(x)
#     print(x.shape)
#     could maybe do sigmoid dense through time. not sure which is better
#     x = Flatten()(x)
#     do I want a pass option? since this isnt competitive I dont think it makes sense
    x = Dense(output_length, activation = "softmax")(x)
    return x

# def value_head(prev):
#     x = Conv1D(1, 1)(prev)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = Flatten()(x)
#     x = Dense(32)(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
# #     in the image it says no batch norm, but it seems like that would be better...
# # going to try adding it
#     x = Dense(1, activation="tanh")(x)
#     return x

# def policy_head(prev):
#     x = Conv1D(1, 1)(prev)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
# #     no batch norm or relu in between, a bit confusing, since this would make them
# # mergeable I think
# # going to try adding them for now
#     x = Conv1D(1, 1)(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
# #     do I want a pass option? since this isnt competitive I dont think it makes sense
#     x = Dense(vector_size, activation = "softmax")(x)
#     return x

def create_net(time_steps, input_length, output_length, num_layers=8):
#     going to try ~halving the history size, since it should be shorter in general

#     vector_size = whatever the flattened total board size is
#     so for example with 10 the number would be 9 + 8 I think

# could try normalizing word vectors, but it will remove the notion of length
# this could be an issue since some words(layer types) will occur more often and should 
# have more weight
# https://stats.stackexchange.com/questions/177905/should-i-normalize-word2vecs-word-vectors-before-using-them

    #skips_input = Input(shape=(time_steps, vector_size))
    #non_skips_input = Input(shape=(time_steps, vector_size))

    inp = Input(shape=(time_steps, input_length))
#     x = conv_layer(x)
    x = inp
    for _ in range(num_layers):
        x = lstm_layer(x)
    policy = lstm_policy_head(x, output_length)
    value = lstm_value_head(x)
#     could try TimeDistributedDense instead of Flatten + Dense + softmax. Not sure which is better

    optim = Nadam()
    base_lr = 0.001
    max_lr = 0.006
    clr = CyclicLR(base_lr=base_lr, max_lr=max_lr,
                   step_size=2000., mode='triangular')

    model.compile(optimizer=optim,
                  loss=AlphaZeroLoss,
                 callbacks=[clr])

    def AlphaZeroLoss(true_value, pred_value):
#     should have l2 regularization
    return K.pow(true_value - pred_value, 2) - K.dot(transpose(mcts_probas), K.log(pred_probas))
         
    
    model = Model(inputs=inp, outputs=[policy, value])
    
    return model



# idea: conv filters selected by an LSTM
# progressively shrinking filters
# goal is to split up an image by it's most important components
# maybe a conv2dlstm which does rotations, flips, and crops, shears, blurs, etc





# idea: send in last 8 game states (maybe 16) or it might be less since there
# are in general less game states in architecture search

# for the rollout net just do a certain size window, maybe 3x3 which would equal
# 9 spaces

# since I am using vectors rather than matrices, I think 1d Convs and LSTM's would be 
# best

# Idea, stochastically drop some part of the graph between the end and the first layer
# (cant drop the first layer)
# this will create more training data
# more ways to try to create more data will be helpful
# can try randomly masking a couple of grid spots and see if that helps improve
# accuracy

# My goal is to make everything a lot smaller so I will be aiming for 1/8 of the size 
# for everything (for now)


# from keras.layers import LSTM, Lambda
# from keras.layers.merge import add

# def make_residual_lstm_layers(input, rnn_width=17, rnn_depth=2, rnn_dropout=.2):
#     """
#     The intermediate LSTM layers return sequences, while the last returns a single element.
#     The input is also a sequence. In order to match the shape of input and output of the LSTM
#     to sum them we can do it only for all layers but the last.
#     """
#     x = input
#     for i in range(rnn_depth):
#         return_sequences = i < rnn_depth - 1
#         x_rnn = Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, 
#                                    return_sequences=return_sequences, activation=None))(x)
#         x_rnn = BatchNormalization()(x_rnn)
#         if return_sequences:
#             # Intermediate layers return sequences, input is also a sequence.
#             if i > 0 or input.shape[-1] == rnn_width:
#                 x = add([x, x_rnn])
#             else:
#                 # Note that the input size and RNN output has to match, due to the sum operation.
#                 # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
#                 x = x_rnn
#         else:
#             # Last layer does not return sequences, just the last element
#             # so we select only the last element of the previous output.
#             def slice_last(x):
#                 return x[..., -1, :]
#             x = add([Lambda(slice_last)(x), x_rnn])
        
#         x = Activation("tanh")(x)
#     return x