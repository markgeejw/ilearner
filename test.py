# Author: Mark Gee
# Platform: keras
# Testing script for ILearner

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
parser.add_argument('--weights', default=None, help="Path to weights to be loaded to start training from (optional).", required=True)
parser.add_argument('--classes', default=2, help="Number of comprehension levels to train for. 2(default), 3 or 6.")
parser.add_argument('--diff', default='easy', help="Passage difficulty to use. 'easy'(default), 'medium', 'hard' or 'all'.")
parser.add_argument('--window', default=60, help="Subsequence window length. Default: 60.")
parser.add_argument('--overlap', default=30, help="Number of frames for overlap of windows. Default: 30.")
parser.add_argument('--filter_type', default='moving_average', help="Filter type to use for data. moving_average(default), kalman or none.")
parser.add_argument('--ma_window', default=10, help="Moving average filter window length. Default: 10.")
parser.add_argument('--gaze_error', default=3, help="Gaze tracking model error to be used for kalman filter. Default: 3.")
parser.add_argument('--stateful', default=False, help="Whether or not to use stateful LSTMs. Default: False.")
args = parser.parse_args()

# Load training parameters
n_classes = args.classes
diff = args.diff
window_size = args.window
overlap = args.overlap
filter_type = args.filter_type
ma_window = args.ma_window
gaze_error = args.gaze_error
stateful = args.stateful

# Default batch sizes for testing is the size of the split
test_batch_size = 6 if not stateful else 1

# Create training and val data generators based on configurations
test_data = ILearnerData.ILearnerData(test_batch_size, window_size, overlap, text_diff=diff, split='test', filter_type=filter_type, ma_window=ma_window, n_classes=n_classes, model_error=gaze_error)

if args.model == 'lstm':
    model = ILearnerModel.lstm(batch_size=test_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
elif args.model == 'gru':
    model = ILearnerModel.gru(batch_size=test_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
# For convolutional models, we use a length size of 15 to compare spatial relationships
elif args.model == 'cnn_lstm':
    model = ILearnerModel.cnn_lstm(15, batch_size=test_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)
elif args.model == 'convlstm2d':
    model = ILearnerModel.convlstm2d(15, batch_size=test_batch_size, n_timesteps=window_size, stateful=stateful, levels=n_classes)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
model.load_weights(args.weights)

# Testing
results = model.evaluate_generator(test_data, verbose=1)
print('Test results: \n', results)