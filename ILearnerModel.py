#%%
from keras.layers import LSTM, GRU, ConvLSTM2D, Dense, Flatten, Dropout, LSTM, TimeDistributed, ConvLSTM2D, Conv1D, MaxPooling1D, Masking, Reshape
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Layer, Input

import numpy as np

def lstm(batch_size=None, n_timesteps=None, stateful=False, levels=6):
    if stateful:
        input = Input(batch_shape=(batch_size, n_timesteps, 8))
    else:
        input = Input(shape=(None, 8))
    x = Masking(mask_value=0.)(input)
    x = LSTM(128, stateful=stateful)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(levels, activation='softmax')(x)
    return Model(input, x)

def gru(batch_size=None, n_timesteps=None, stateful=False, levels=6):
    if stateful:
        input = Input(batch_shape=(batch_size, n_timesteps, 8))
    else:
        input = Input(shape=(None, 8))
    x = Masking(mask_value=0.)(input)
    x = GRU(256, stateful=stateful)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(levels, activation='softmax')(x)
    return Model(input, x)
    
def cnn_lstm(n_length, batch_size=None, n_timesteps=None, stateful=False, levels=6):
    if stateful:
        input = Input(batch_shape=(batch_size, n_timesteps, 8))
    else:
        input = Input(shape=(None, 8))
    x = Reshape((n_timesteps//n_length, n_length, 8))(input)
    x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, stateful=stateful)(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(levels, activation='softmax')(x)
    return Model(input, x)

def convlstm2d(n_length, batch_size=None, n_timesteps=None, stateful=False, levels=6):
    if stateful:
        input = Input(batch_shape=(batch_size, n_timesteps, 8))
    else:
        input = Input(shape=(None, 8))
    x = Reshape((n_timesteps//n_length, 1, n_length, 8))(input)
    x = ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_timesteps//n_length, 1, n_length, 8), stateful=stateful)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(levels, activation='softmax')(x)
    return Model(input, x)


#%%
