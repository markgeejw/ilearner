
#%%
import os
os.chdir('/home_nfs/markgee/gaze-extract')
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))
#%%
import scipy.io as sio
import numpy as np
from keras.optimizers import Adam

from utils import ResetStatesCallback, f1

#%%
import ILearnerModel
import ILearnerData
#%%
epochs = 20
n_classes = [6]
levels = ['easy']
window_size = 60
overlap = 30
ma_window = 10

# #%%

# #%%
# for level in levels:
#     for n_class in n_classes:
#         train_data = ILearnerData.ILearnerData(11, window_size, overlap, text_diff=level, filter_type=ma_window, n_classes=n_class)
#         val_data = ILearnerData.ILearnerData(5, window_size, overlap, text_diff=level, split='val', ma_window=ma_window, n_classes=n_class)
#         reset_state_callback = ResetStatesCallback(train_data.n_sequences)
#         accuracy = []
#         val_accuracy = []
#         f1_val = []
#         val_f1 = []
#         model = ILearnerModel.gru(11, n_timesteps=window_size, stateful=False, levels=n_class)
#         model.summary()
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
#         for epoch in range(epochs):
#             history = model.fit_generator(generator=train_data,
#                                     epochs=1,
#                                     verbose=1,
#                                     validation_data=val_data)
#             accuracy = np.append(accuracy, history.history['acc'])
#             val_accuracy = np.append(val_accuracy, history.history['val_acc'])
#             f1_val = np.append(f1_val, history.history['f1'])
#             val_f1 = np.append(val_f1, history.history['val_f1'])
#             np.save('gru/gru_accuracy.npy', accuracy)
#             np.save('gru/gru_val_accuracy.npy', val_accuracy)
#             np.save('gru/gru_f1.npy', f1_val)
#             np.save('gru/gru_val_f1.npy', val_f1)
#             model.save_weights('gru/gru_%d.h5' % (epoch+1))
        
#         accuracy = []
#         val_accuracy = []
#         f1_val = []
#         val_f1 = []
#         model = ILearnerModel.cnn_lstm(30, batch_size=11, n_timesteps=window_size, stateful=False, levels=n_class)
#         model.summary()
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
#         for epoch in range(epochs):
#             history = model.fit_generator(generator=train_data,
#                                     epochs=1,
#                                     verbose=1,
#                                     validation_data=val_data)
#             accuracy = np.append(accuracy, history.history['acc'])
#             val_accuracy = np.append(val_accuracy, history.history['val_acc'])
#             f1_val = np.append(f1_val, history.history['f1'])
#             val_f1 = np.append(val_f1, history.history['val_f1'])
#             np.save('cnn_lstm/cnn_lstm_accuracy.npy', accuracy)
#             np.save('cnn_lstm/cnn_lstm_val_accuracy.npy', val_accuracy)
#             np.save('cnn_lstm/cnn_lstm_f1.npy', f1_val)
#             np.save('cnn_lstm/cnn_lstm_val_f1.npy', val_f1)
#             model.save_weights('cnn_lstm/cnn_lstm_%d.h5' % (epoch+1))
        
#         accuracy = []
#         val_accuracy = []
#         f1_val = []
#         val_f1 = []
#         model = ILearnerModel.convlstm2d(30, batch_size=11, n_timesteps=window_size, stateful=False, levels=n_class)
#         model.summary()
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
#         for epoch in range(epochs):
#             history = model.fit_generator(generator=train_data,
#                                     epochs=1,
#                                     verbose=1,
#                                     validation_data=val_data)
#             accuracy = np.append(accuracy, history.history['acc'])
#             val_accuracy = np.append(val_accuracy, history.history['val_acc'])
#             f1_val = np.append(f1_val, history.history['f1'])
#             val_f1 = np.append(val_f1, history.history['val_f1'])
#             np.save('convlstm2d/convlstm2d_accuracy.npy', accuracy)
#             np.save('convlstm2d/convlstm2d_val_accuracy.npy', val_accuracy)
#             np.save('convlstm2d/convlstm2d_f1.npy', f1_val)
#             np.save('convlstm2d/convlstm2d_val_f1.npy', val_f1)
#             model.save_weights('convlstm2d/convlstm2d_%d.h5' % (epoch+1))

#%%
test_data = ILearnerData.ILearnerData(6, window_size, overlap, text_diff='easy', split='test', ma_window=ma_window, n_classes=2)
model = ILearnerModel.lstm(batch_size=6, n_timesteps=window_size, stateful=False, levels=2)
model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
# model.load_weights('./convlstm2d/convlstm2d_18.h5')
# print(model.evaluate_generator(test_data))

#%%
# easy: 11, 16, 20
# medium: 4, 9, 4
# hard: 5, 12, 20
# all: 14, 13, 3