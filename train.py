# Author: Mark Gee
# Platform: keras
# Training script for ILearner

import scipy.io as sio
import numpy as np
from keras.optimizers import Adam

# Import model and data generator modules
import ILearnerModel
import ILearnerData

# Import helper functions from utils
from utils import ResetStatesCallback, f1

import argparse

parser = argparse.ArgumentParser(description='Training the ILearner')
parser.add_argument('--model', default='lstm', help="Model to use (lstm(default), gru, cnn_lstm, convlstm2d).")
parser.add_argument('--epochs', default=1, help="Number of epochs to train (default: 1)")
parser.add_argument('--classes', default=2, help="Number of comprehension levels to train for. 2(default), 3 or 6.")
parser.add_argument('--diff', default='easy', help="Passage difficulty to use. 'easy'(default), 'medium', 'hard' or 'all'.")
parser.add_argument('--window', default=60, help="Subsequence window length. Default: 60.")
parser.add_argument('--overlap', default=30, help="Number of frames for overlap of windows. Default: 30.")
parser.add_argument('--filter_type', default='moving_average', help="Filter type to use for data. moving_average(default), kalman or none.")
parser.add_argument('--ma_window', default=10, help="Moving average filter window length. Default: 10.")
parser.add_argument('--gaze_error', default=3, help="Gaze tracking model error to be used for kalman filter. Default: 3.")
parser.add_argument('--stateful', default=False, help="Whether or not to use stateful LSTMs. Default: False.")
parser.add_argument('--weights', default=None, help="Path to weights to be loaded to start training from (optional).")
args = parser.parse_args()

# Load training parameters
epochs = args.epochs
n_classes = args.classes
diff = args.diff
window_size = args.window
overlap = args.overlap
filter_type = args.filter_type
ma_window = args.ma_window
gaze_error = args.gaze_error
stateful = args.stateful

# Default batch sizes for training is the size of the split
train_batch_size = 11 if not stateful else 1
val_batch_size = 5 if not stateful else 1

# Create training and val data generators based on configurations
train_data = ILearnerData.ILearnerData(train_batch_size, window_size, overlap, text_diff=diff, filter_type=filter_type, ma_window=ma_window, n_classes=n_classes, model_error=gaze_error)
val_data = ILearnerData.ILearnerData(val_batch_size, window_size, overlap, text_diff=diff, split='val', filter_type=filter_type, ma_window=ma_window, n_classes=n_classes, model_error=gaze_error)

# Create callback function that resets states if using stateful
if stateful:
    reset_state_callback = ResetStatesCallback(train_data.n_sequences)

if args.model == 'lstm':
    model = ILearnerModel.lstm(batch_size=train_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
elif args.model == 'gru':
    model = ILearnerModel.gru(batch_size=train_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
# For convolutional models, we use a length size of 15 to compare spatial relationships
elif args.model == 'cnn_lstm':
    model = ILearnerModel.cnn_lstm(15, batch_size=train_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
elif args.model == 'convlstm2d':
    model = ILearnerModel.convlstm2d(15, batch_size=train_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

# Load weights to train from
if args.weights:
    model.load_weights(args.weights)

# Method for tracking accuracy and F1 scores
# Commented out here
# Use your favourite!
# accuracy = []
# val_accuracy = []
# f1_val = []
# val_f1 = []

# Training
for epoch in range(epochs):
    history = model.fit_generator(generator=train_data,
                            epochs=1,
                            verbose=1,
                            callbacks=[reset_state_callback if stateful else None],
                            validation_data=val_data)
    # Store accuracy and F1 scores
    # accuracy = np.append(accuracy, history.history['acc'])
    # val_accuracy = np.append(val_accuracy, history.history['val_acc'])
    # f1_val = np.append(f1_val, history.history['f1'])
    # val_f1 = np.append(val_f1, history.history['val_f1'])
    # Save the stored values as .npy file
    # np.save('gru/gru_accuracy.npy', accuracy)
    # np.save('gru/gru_val_accuracy.npy', val_accuracy)
    # np.save('gru/gru_f1.npy', f1_val)
    # np.save('gru/gru_val_f1.npy', val_f1)
    # Save the model weights
    # model.save_weights('gru/gru_%d.h5' % (epoch+1))

# Save model weights
model.save_weights('models/output/ilearner.h5')