from keras.callbacks import Callback
from pykalman import KalmanFilter
import numpy as np
from keras import backend as K

class ResetStatesCallback(Callback):
    def __init__(self, n_sequences):
        self.counter = 0
        self.n_sequences = n_sequences

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.n_sequences == 0:
            self.model.reset_states()
        self.counter += 1

#moving average filter
def moving_average_filter(data, window):
    N = window
    cumsum, filtered_data = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            filtered_data.append(moving_ave)
    return np.array(filtered_data)

# Kalman filter
def kalman_filter(data, error):
    data=np.ma.masked_less(data,-900)
    Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
    Observation_Matrix=[[1,0,0,0],[0,1,0,0]]
    xinit=data[0,0]
    yinit=data[0,1]
    vxinit=data[1,0]-data[0,0]
    vyinit=data[1,1]-data[0,1]
    initstate=[xinit,yinit,vxinit,vyinit]
    initcovariance=error*np.eye(4)
    transistionCov=error*np.eye(4)
    observationCov=error*np.eye(2)
    kf=KalmanFilter(transition_matrices=Transition_Matrix,
                observation_matrices =Observation_Matrix,
                initial_state_mean=initstate,
                initial_state_covariance=initcovariance,
                transition_covariance=transistionCov,
                observation_covariance=observationCov)
    (filtered_state_means, _) = kf.filter(data)
    return np.array(filtered_state_means)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))