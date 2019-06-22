import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
set_session(tf.Session(config=config))

import cv2
import numpy as np
import scipy.io as sio

from pymediainfo import MediaInfo

import json

# Define path to JSON files in dataset
path_to_json = './data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# Define function to check rotation of video files
# Some mobile devices record in a particular rotation and save the orientation to a metadata
# Video file has to be rotated before frame can be used to extract gaze
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    media_info = MediaInfo.parse(path_video_file)
    for track in media_info.tracks:
        if track.track_type == 'Video':
            rotation = int(float(track.to_data()['rotation']))

    # returns a rotate code that can be used to correct the rotation
    rotateCode = None
    if rotation == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 180:
        rotateCode = cv2.ROTATE_180
    elif rotation == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

# Function to correct the rotation using rotate code
def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 


# face extrator using SFD provided from FAN
# https://github.com/1adrianb/face-alignment
from detection.sfd import FaceDetector

sfd = FaceDetector(device='cuda:0', verbose=True)

#%%
# gaze extractor from gaze tracker
from gazetracker import SEITracker
model = SEITracker.SEITracker()
model.load_weights('./gazetracker/seresnet_30.h5')

# Helper functions to preprocess the image before using gaze tracker
# Crop the image to the bounding boxes
def cropImage(img, bbox):
    bbox = bbox.astype(int)
    x1, y1, x2, y2, _ = bbox
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    return img[y1: y2, x1: x2]

# Obtain 25x25 face grid from bounding boxes
def getFaceGrid(img, bbox):
    x1, y1, x2, y2, _ = bbox
    gridwidth = np.shape(img)[1] / 25
    gridheight = np.shape(img)[0] / 25
    faceGrid = np.zeros((25, 25))
    for y in range(25):
        for x in range(25):
            if (x * gridwidth < x1 or x * gridwidth > x2 or y * gridheight < y1 or y * gridheight > y2):
                faceGridValue = 0
            else:
                faceGridValue = 1
            faceGrid[y][x] = faceGridValue
    return faceGrid

# Obtain face mean to normalize face image as this was how the models were trained
faceMean = sio.loadmat('gazetracker/mean_face_224.mat')['image_mean']

# Arrays to store the appropriate data
easy_scores = []
medium_scores = []
hard_scores = []
easy_gaze_all = []
medm_gaze_all = []
hard_gaze_all = []
fps = []
dims = []

# Loop through all video files using the json details
for i, json_file in enumerate(json_files):
    print('[%d/%d] Processing video (%.2f%%)' % ((i+1), len(json_files), (i+1) / len(json_files) * 100))
    # get metadata
    with open(path_to_json + json_file) as f:
        metadata = json.load(f)
    
    # append scores
    scores = metadata['score']
    easy_gaze = []
    medm_gaze = []
    hard_gaze = []
    easy_scores.append(scores[0])
    medium_scores.append(scores[1])
    hard_scores.append(scores[2])

    # append dims
    dims.append([metadata['width'], metadata['height']])

    # get video file
    video_name = metadata['id'] + '.mp4'
    # open with opencv
    vidcap = cv2.VideoCapture(path_to_json + video_name)
    # obtain rotate code
    rotateCode = check_rotation(path_to_json + video_name)
    # obtain frame rate
    frame_rate = int(vidcap.get(5))
    for _ in range(3):
        fps.append(frame_rate)

    # create an array for frameCount intervals at which the passage was being read
    passageIntervals = []
    for passageCount, passageTime in enumerate(metadata['passageTimes']):
        # do not include last time count which refers to total time taken for exercise
        if passageCount == 3:
            break
        # the passage finishes when the first question of the passage is being done
        # multiply by frame rate to get frame count instead of time
        passageIntervals.append([passageTime * frame_rate, metadata['questionTimes'][passageCount*5] * frame_rate]) 

    # loop over each passage
    for passageCount, passageInterval in enumerate(passageIntervals):
        startFrame, endFrame = passageInterval
        print('Passage: %d/3' % (passageCount+1), '-- Start: ', startFrame, ', End: ', endFrame)
        vidcap.set(1, startFrame)
        while True:
            success, frame = vidcap.read()
            # correct the video's rotation
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)
            # in case of error in retrieving frame
            if frame is None:
                continue
            else:
                image = frame[...,::-1]                                 # convert BGR channels into RGB
                detected_faces = sfd.detect_from_image(image)           # detect face using SFD
                # if no face, skip
                if len(detected_faces) == 0:
                    continue
                else:
                    face_bbox = detected_faces[0]                           # assume first face is user face
                    face_image = cropImage(image, face_bbox)                # obtain face image
                    face_input = cv2.resize(face_image, (224, 224))         # resize face for input to model
                    face_input = face_input - faceMean
                    face_grid = getFaceGrid(image, face_bbox)               # obtain face grid for input to model
                    # extract gaze here
                    gaze = model.predict({'face': np.expand_dims(face_input,0), 'grid': np.reshape(face_grid, (1,625))}).flatten()
            if passageCount == 0:
                easy_gaze.append(gaze)
            elif passageCount == 1:
                medm_gaze.append(gaze)
            else:
                hard_gaze.append(gaze)
            # Finished looping over relevant frames, so break
            if startFrame == endFrame:
                break
            # Move onto next frame
            startFrame += 1
    easy_gaze_all.append(easy_gaze)
    medm_gaze_all.append(medm_gaze)
    hard_gaze_all.append(hard_gaze)

gazereader_dataset = {  'easy_gaze': easy_gaze_all, 
                        'medium_gaze': medium_gaze_all,
                        'hard_gaze': hard_gaze_all,
                        'easy_scores': easy_scores,
                        'medium_scores': medium_scores,
                        'hard_scores': hard_scores,
                        'fps': fps,
                        'dims': dims }

# Save the extracted data to a .mat file
sio.savemat('./gazereader_data.mat', gazereader_dataset)
