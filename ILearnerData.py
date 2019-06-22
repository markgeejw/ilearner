# Author: Mark Gee
# Platform: keras
# Data generator for ILearner
# Load from extracted gaze coordinates
# Executes filtering, windowing, etc.

import scipy.io as sio
import numpy as np
from keras.utils import to_categorical, Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback

import random

from utils import moving_average_filter, kalman_filter

DATASET_PATH = './gazereader_data.mat'

class ILearnerData(Sequence):
    def __init__(self, batch_size, 
                    window_size, 
                    overlap, 
                    split='train', 
                    filter_type='moving_average',
                    text_diff='easy',
                    ma_window=10,
                    model_error=None,
                    n_classes=6, 
                    random_data=False, 
                    include_low_fps=False):
        self.batch_size = batch_size
        self.window_size = window_size
        self.overlap = overlap
        self.n_classes = n_classes
        self.random = random_data

        # Load dataset into arrays 
        dataset = sio.loadmat(DATASET_PATH)
        if text_diff == 'all':
            gaze_coordinates_all = np.concatenate([dataset['easy_gaze'], dataset['medium_gaze'], dataset['hard_gaze']], axis=1)[0]
            raw_scores = np.concatenate([dataset['easy_scores'], dataset['medium_scores'], dataset['hard_scores']], axis=1)[0]
            fps = np.array([dataset['fps']] * 3).reshape((90))
            dims = np.array([dataset['dims']] * 3).reshape((90, 2))
        else:
            gaze_coordinates_all = dataset['%s_gaze' % text_diff][0]
            raw_scores = dataset['%s_scores' % text_diff][0]
            fps = dataset['fps'][0]
            dims = dataset['dims']
        scores = to_categorical(raw_scores // (6 / n_classes), num_classes=n_classes)

        # Total number of samples
        self.n_samples = gaze_coordinates_all.shape[0]
        
        # Normalize the gaze coordinates to screen size
        # Multiply by 100 so the coordinates are not too small and more intuitive (ranges around -1 to 1)
        # We also index only 30 frames after the start and 30 frames before the end, ensuring reader has started 
        # to read before obtaining the gaze
        gaze_coordinates_all = [gaze_coordinates[30:-30] / dims[i] * 100 for i,gaze_coordinates in enumerate(gaze_coordinates_all)]
        # Filter the data using moving average or kalman
        gaze_coordinates_all = self.filter_data(gaze_coordinates_all, type=filter_type, window=ma_window, error=model_error)
        # Augment data into {x, y, x_dist, y_dist, total_dist, x_vel, y_vel, total_vel}
        gaze_coordinates_all = self.augment_data(gaze_coordinates_all, fps)
        # Create windowed data into order to use stateful lstms to apply tbppt for long sequence data
        self.n_sequences, gaze_data, score_data = self.create_windowed_data(gaze_coordinates_all, scores)
        # Reshape data such that each batch retrieves from the same sample until all subsequences are trained on
        self.gaze_data = np.reshape(gaze_data, (self.n_samples, -1, 8))
        self.score_data = np.reshape(score_data, (self.n_samples, -1, n_classes))

        # Create indices to index each sample 
        if not include_low_fps:
            # Filter out low fps samples
            # Note that we could have filtered out low fps at the earlier stage
            # to skip preprocessing of filtered out samples but this is
            # simpler to implement and the time saved is insignificant
            indices = np.argwhere(dataset['fps'][0] > 27).squeeze()
        else:
            indices = np.arange(0, self.n_samples)

        # Update n_samples
        self.n_samples = len(indices)
        # Apply 50-25-25 split
        n_train = int(self.n_samples * 0.5)
        n_val = int(self.n_samples * 0.25)

        if split == 'train':
            indices = indices[:n_train]
        elif split == 'val':
            indices = indices[n_train:n_train+n_val]
        else:
            indices = indices[n_train+n_val:]
        
        # If using all text difficulties, expand indices to index easy, medium, hard samples
        # We use all samples here
        if text_diff == 'all':
            indices = np.concatenate([indices, indices+30, indices+60])
        self.indices = indices

        # Shuffle the indices before training
        random.shuffle(self.indices)

    def filter_data(self, gaze_coordinates_all, type='moving_average', window=10, error=None):
        '''Applies filters to each sample's gaze coordinates'''
        if type == 'moving_average':
            return np.array([moving_average_filter(gaze_coordinates, window) for gaze_coordinates in gaze_coordinates_all])
        
        elif type == 'kalman':
            # Kalman filter requires the model error
            if error == None:
                raise Exception('Kalman filter requires the model error!')
            return np.array([kalman_filter(gaze_coordinates, error) for gaze_coordinates in gaze_coordinates_all])
        else:
            return gaze_coordinates_all

    def augment_data(self, gaze_coordinates_all, fps):
        '''Augment the dataset to include distance and velocity in each axis'''
        # Apply to each sample
        for sample, gaze_coordinates in enumerate(gaze_coordinates_all):
            # Obtain fps and period between each frame to calculate velocity
            time_per_sample = 1/fps[sample]
            # Create an array of gaze coordinates shifted one sample forward
            # First sample is repeated to allow initial speed to be 0
            prev_data = np.insert(gaze_coordinates[:-1], 0, gaze_coordinates[0], axis=0)
            # Calculate distance x and y
            dist_data = gaze_coordinates - prev_data
            # Calculate overall distance using norm
            overall_dist_data =  np.expand_dims(np.linalg.norm(dist_data, axis=-1), -1)
            # Calculate speeds by diving by time between frames
            speed_data = dist_data/time_per_sample
            overall_speed_data = overall_dist_data/time_per_sample
            # Augment data
            augmented_data = np.concatenate([gaze_coordinates, dist_data, overall_dist_data, speed_data, overall_speed_data], axis=-1)
            gaze_coordinates_all[sample] = augmented_data
        return gaze_coordinates_all

    def create_windowed_data(self, gaze_coordinates_all, scores):
        '''Create windowed data with overlaps'''
        # Calculate the maximum length of sequences in dataset for pad length
        max_sequence_length = max([sequence_length for sequence_length in [np.shape(coordinates)[0] for coordinates in gaze_coordinates_all]])
        # Calculate the padding length by first subtracting from the total length the window size
        # The remaining sequence length should be divisible by the overlap inverse (window_size - overlap) since the window shifts
        # by overlap_inverse for each subsequence
        overlap_inverse = self.window_size - self.overlap
        padding_length = self.window_size + overlap_inverse * ((max_sequence_length - self.window_size) // overlap_inverse + 1)
        # Pad the sequences with an unused value of 0 to be Masked when training
        gaze_coordinates_padded = pad_sequences(gaze_coordinates_all, maxlen=padding_length, dtype='float32', padding='post', value=0.)
        # Create windowed data
        gaze_coordinates_windowed = np.array([self.rolling_window(gaze_coordinates) for gaze_coordinates in gaze_coordinates_padded])
        n_sequences = np.shape(gaze_coordinates_windowed)[1]
        scores_windowed = np.array([[score] * n_sequences for score in scores])
        return n_sequences, gaze_coordinates_windowed, scores_windowed

    def rolling_window(self, input):
        '''Function to create windowed data'''
        # Continually index the data by shifting, slicing and appending
        output = []
        index = 0
        # Shift by overlap_inverse each time
        overlap_inverse = self.window_size - self.overlap
        while index < input.shape[0] - self.window_size:
            # Obtain appropriate slice of data
            window = input[index:index+self.window_size]
            # Append to output
            output.append(window)
            # Shift
            index += overlap_inverse
        return output

    def __getitem__(self, index):
        # Calculate the current batch number based on index, since shuffle will be turned off
        batch_num = index // self.n_sequences
        # Calculate current subsequence number
        seq_num = index % self.n_sequences
        # Placeholder for data output
        gaze_batch = np.empty((self.batch_size, self.window_size, 8))
        score_batch = np.empty((self.batch_size, self.n_classes))
        # Retrieve each data for each sample in batch
        for i in range(self.batch_size):
            idx = self.indices[batch_num*self.batch_size+i]
            gaze_sequence = self.gaze_data[idx][seq_num*self.window_size:(seq_num+1)*self.window_size]
            gaze_batch[i] = gaze_sequence
            score_batch[i] = self.score_data[idx][seq_num]
        return gaze_batch, score_batch
    
    def __len__(self):
        return int(len(self.indices) * self.n_sequences / self.batch_size) 
    
    def on_epoch_end(self):
        random.shuffle(self.indices)



#%%
