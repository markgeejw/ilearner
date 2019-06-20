#%%
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

import os
import json

#%%
os.chdir('/home_nfs/markgee/gaze-extract')

#%%
path_to_json = '/home_nfs/markgee/gaze-extract/data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

#%%
# List of scores for each passage difficulty
easy_score = []
medium_score = []
hard_score = []

# List of durations for each passage difficulty
easy_dur = []
medium_dur = []
hard_dur = []

# Number of participants who scored 0, 1, 2, 3, 4, 5 respectively
score_classes = [0, 0, 0, 0, 0, 0]

#%%
# fps = []
# dims = []
for i, json_file in enumerate(json_files):
    if i in [5, 7, 8, 12, 15, 17, 23, 29]:
        continue
    # get metadata
    with open(path_to_json + json_file) as f:
        metadata = json.load(f)

    # print('Width: ', int(metadata['width']), ' Height: ', int(metadata['height']))
    # widths.append(int(metadata['width']))
    # heights.append(int(metadata['height']))
    # dims.append([int(metadata['width']), int(metadata['height'])])

    scores = metadata['score']
    for score in scores:
        score_classes[score] += 1
    easy_score.append(scores[0])
    medium_score.append(scores[1])
    hard_score.append(scores[2])
    easy_dur.append(metadata['questionTimes'][0] - metadata['passageTimes'][0])
    medium_dur.append(metadata['questionTimes'][5] - metadata['passageTimes'][1])
    hard_dur.append(metadata['questionTimes'][10] - metadata['passageTimes'][2])

# print('Easy score: Mean: %.03f, Std: %.03f' % (np.mean(easy_score), np.std(easy_score)))
# print('Medium score: Mean: %.03f, Std: %.03f' % (np.mean(medium_score), np.std(medium_score)))
# print('Hard score: Mean: %.03f, Std: %.03f' % (np.mean(hard_score), np.std(hard_score)))

print('Easy duration: Mean: %.03f, Std: %.03f' % (np.mean(easy_dur), np.std(easy_dur)))
print('Medium duration: Mean: %.03f, Std: %.03f' % (np.mean(medium_dur), np.std(medium_dur)))
print('Hard duration: Mean: %.03f, Std: %.03f' % (np.mean(hard_dur), np.std(hard_dur)))

# print('Correlation: ', np.corrcoef(easy_score+medium_score+hard_score, easy_dur+medium_dur+hard_dur)[0, 1])
print('Score classes: ', score_classes)

#%%
longest_dur = 0
shortest_dur = 999
for dur in easy_dur + medium_dur + hard_dur:
    if dur > longest_dur:
        longest_dur = dur
    if dur < shortest_dur:
        shortest_dur = dur

#%%
data = sio.loadmat('./gazereader_dims.mat')
easy_gaze = data['easy_gaze'][0][0]

#%%
plt.style.use('classic')

def plot_gaze(data):
    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.5)
    fig.set_figwidth(15)
    ax[0].plot(data[:,0])
    # ax[0].set_ylim(-5,5)
    ax[0].set_title('Gaze Coordinates X (Original)')
    ax[0].set_xlabel('Timestep')
    ax[0].set_ylabel('X-Coordinate')
    ax[1].plot(data[:,1])
    # ax[1].set_ylim(-10,0)
    ax[1].set_title('Gaze Coordinates Y (Original)')
    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel('Y-Coordinate')


#%%
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

#%%
# Kalman filter
def kalman_filter(data,error):
    data=np.ma.masked_less(data,-500)
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
    return filtered_state_means

#%%
# plot curves
import numpy as np

acc = []
easy_lstm_acc_6 = np.load('./lstm/easy/6/lstm_accuracy.npy')
acc.append(easy_lstm_acc_6)
print('Easy LSTM-6: \n', easy_lstm_acc_6)
easy_lstm_acc_3 = np.load('./lstm/easy/3/lstm_accuracy.npy')
acc.append(easy_lstm_acc_3)
print('Easy LSTM-3: \n', easy_lstm_acc_3)
easy_lstm_acc_2 = np.load('./lstm/easy/2/lstm_accuracy.npy')
acc.append(easy_lstm_acc_2)
print('Easy LSTM-2: \n', easy_lstm_acc_2)
medium_lstm_acc_6 = np.load('./lstm/medium/6/lstm_accuracy.npy')
acc.append(medium_lstm_acc_6)
print('medium LSTM-6: \n', medium_lstm_acc_6)
medium_lstm_acc_3 = np.load('./lstm/medium/3/lstm_accuracy.npy')
acc.append(medium_lstm_acc_3)
print('medium LSTM-3: \n', medium_lstm_acc_3)
medium_lstm_acc_2 = np.load('./lstm/medium/2/lstm_accuracy.npy')
acc.append(medium_lstm_acc_2)
print('medium LSTM-2: \n', medium_lstm_acc_2)
hard_lstm_acc_6 = np.load('./lstm/hard/6/lstm_accuracy.npy')
acc.append(hard_lstm_acc_6)
print('hard LSTM-6: \n', hard_lstm_acc_6)
hard_lstm_acc_3 = np.load('./lstm/hard/3/lstm_accuracy.npy')
acc.append(hard_lstm_acc_3)
print('hard LSTM-3: \n', hard_lstm_acc_3)
hard_lstm_acc_2 = np.load('./lstm/hard/2/lstm_accuracy.npy')
acc.append(hard_lstm_acc_2)
print('hard LSTM-2: \n', hard_lstm_acc_2)
all_lstm_acc_6 = np.load('./lstm/all/6/lstm_accuracy.npy')
acc.append(all_lstm_acc_6)
print('all LSTM-6: \n', all_lstm_acc_6)
all_lstm_acc_3 = np.load('./lstm/all/3/lstm_accuracy.npy')
acc.append(all_lstm_acc_3)
print('all LSTM-3: \n', all_lstm_acc_3)
all_lstm_acc_2 = np.load('./lstm/all/2/lstm_accuracy.npy')
acc.append(all_lstm_acc_2)
print('all LSTM-2: \n', all_lstm_acc_2)

#%%
val_acc = []
easy_lstm_val_acc_6 = np.load('./lstm/easy/6/lstm_val_accuracy.npy')
val_acc.append(easy_lstm_val_acc_6)
print('Easy LSTM-6: \n', easy_lstm_val_acc_6)
easy_lstm_val_acc_3 = np.load('./lstm/easy/3/lstm_val_accuracy.npy')
val_acc.append(easy_lstm_val_acc_3)
print('Easy LSTM-3: \n', easy_lstm_val_acc_3)
easy_lstm_val_acc_2 = np.load('./lstm/easy/2/lstm_val_accuracy.npy')
val_acc.append(easy_lstm_val_acc_2)
print('Easy LSTM-2: \n', easy_lstm_val_acc_2)
medium_lstm_val_acc_6 = np.load('./lstm/medium/6/lstm_val_accuracy.npy')
val_acc.append(medium_lstm_val_acc_6)
print('medium LSTM-6: \n', medium_lstm_val_acc_6)
medium_lstm_val_acc_3 = np.load('./lstm/medium/3/lstm_val_accuracy.npy')
val_acc.append(medium_lstm_val_acc_3)
print('medium LSTM-3: \n', medium_lstm_val_acc_3)
medium_lstm_val_acc_2 = np.load('./lstm/medium/2/lstm_val_accuracy.npy')
val_acc.append(medium_lstm_val_acc_2)
print('medium LSTM-2: \n', medium_lstm_val_acc_2)
hard_lstm_val_acc_6 = np.load('./lstm/hard/6/lstm_val_accuracy.npy')
val_acc.append(hard_lstm_val_acc_6)
print('hard LSTM-6: \n', hard_lstm_val_acc_6)
hard_lstm_val_acc_3 = np.load('./lstm/hard/3/lstm_val_accuracy.npy')
val_acc.append(hard_lstm_val_acc_3)
print('hard LSTM-3: \n', hard_lstm_val_acc_3)
hard_lstm_val_acc_2 = np.load('./lstm/hard/2/lstm_val_accuracy.npy')
val_acc.append(hard_lstm_val_acc_2)
print('hard LSTM-2: \n', hard_lstm_val_acc_2)
all_lstm_val_acc_6 = np.load('./lstm/all/6/lstm_val_accuracy.npy')
val_acc.append(all_lstm_val_acc_6)
print('all LSTM-6: \n', all_lstm_val_acc_6)
all_lstm_val_acc_3 = np.load('./lstm/all/3/lstm_val_accuracy.npy')
val_acc.append(all_lstm_val_acc_3)
print('all LSTM-3: \n', all_lstm_val_acc_3)
all_lstm_val_acc_2 = np.load('./lstm/all/2/lstm_val_accuracy.npy')
val_acc.append(all_lstm_val_acc_2)
print('all LSTM-2: \n', all_lstm_val_acc_2)

#%%
acc = []
# lstm_acc = np.load('./lstm/lstm_6_accuracy.npy')
# acc.append(lstm_acc)
# print('LSTM-6: \n', lstm_acc)
gru_acc = np.load('./gru/gru_6_accuracy.npy')
acc.append(gru_acc)
print('GRU-6: \n', gru_acc)
cnn_lstm_acc = np.load('./cnn_lstm/cnn_lstm_6_accuracy.npy')
acc.append(cnn_lstm_acc)
print('CNN-LSTM-6: \n', cnn_lstm_acc)
convlstm_acc = np.load('./convlstm2d/convlstm2d_6_accuracy.npy')
acc.append(convlstm_acc)
print('ConvLSTM-6: \n', convlstm_acc)

#%%
val_acc = []
# lstm_val_acc = np.load('./lstm/lstm_6_val_accuracy.npy')
# val_acc.append(lstm_val_acc)
# print('LSTM-6: \n', lstm_val_acc)
gru_val_acc = np.load('./gru/gru_6_val_accuracy.npy')
val_acc.append(gru_val_acc)
print('GRU-6: \n', gru_val_acc)
cnn_lstm_val_acc = np.load('./cnn_lstm/cnn_lstm_6_val_accuracy.npy')
val_acc.append(cnn_lstm_val_acc)
print('CNN-LSTM-6: \n', cnn_lstm_val_acc)
convlstm_val_acc = np.load('./convlstm2d/convlstm2d_6_val_accuracy.npy')
val_acc.append(convlstm_val_acc)
print('ConvLSTM-6: \n', convlstm_val_acc)

#%%
models = ["Easy-6", "Easy-3", "Easy-2", "Medium-6", "Medium-3", "Medium-2", "Hard-6", "Hard-3", "Hard-2", "All-6", "All-3", "All-2"]
#%%
import matplotlib.pyplot as plt

plt.style.use('classic')

#%%
fig, axes = plt.subplots(4, 3)
row = 0
col = 0
fig.subplots_adjust(hspace=0.5)
fig.set_figwidth(15)
fig.set_figheight(10)
for model in range(len(acc)):
    axes[model//3][model%3].plot(acc[model])
    axes[model//3][model%3].plot(val_acc[model])
    axes[model//3][model%3].legend(['Train', 'Val'])
    axes[model//3][model%3].set_title(models[model])
    axes[model//3][model%3].set_ylabel('Accuracy')
    axes[model//3][model%3].set_xlabel('Epoch')
    axes[model//3][model%3].set_ylim(top=1, bottom=0)

#%%
