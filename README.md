# ILearner

Deep learning model for inferring comprehension from gaze in mobile learning environments. Created for a final year project. Created using Keras. Dataset is required to run the scripts.

The premise is that we can create cheap [eye trackers](https://github.com/markgeejw/gazetracker) using only mobile device cameras. While inaccurate, reading patterns tend to be very regular and predictable. Hence a deep learning model that learns features from noisy gaze data should still function well for reading tasks. Consequently, using only gaze data, we were able to obtain excellent accuracy in classifying comprehension levels. We did this for three different comprehension levels, and for easy passages we have the following results:

| Number of comprehension levels     | 2             | 3             | 6             |
| -----------------------------------|---------------|---------------|---------------|
| Accuracy                           | 0.827         | 0.811         | 0.832         |
| F1                                 | 0.827         | 0.810         | 0.373         |

The results for easy passages are encouraging but poorer for medium and hard passages. The reasons are mainly analyzed in the report but we attribute it to the data collection procedure. We believe that with a larger dataset that can be collected with GazeReader, the model here can be used reliably, even with inaccurate but cheap (free since using mobile device camera) gaze trackers.

## Dataset

The dataset was collected using the mobile application [GazeReader](https://github.com/markgee/GazeReader). We then used the gaze tracker implemented [here](https://github.com/markgee/gazetracker) to obtain the gaze coordinates. An example of the gaze extraction procedure can be seen in the script `preprocess_gaze.py`. This example uses the SE-ResNet-ITracker, trained after 30 epochs to extract gaze. If you want to use your own, modify the script to point to your own model.

You can use the pre-processed data `gazereader_data.mat` to begin training or testing.

The jupyter notebook for testing the dataset statistics are included in `test/Test.ipynb`.

## Installing dependencies

Install the required dependencies using:

```shell
pip install -r requirements.txt
```

## Training

You can execute training by running the following command:

```shell
python train.py --model <MODEL_TO_USE> --epochs <EPOCHS> [OPTIONS]

Options:
--model             Model to use (lstm(default), gru, cnn_lstm, convlstm2d)
--epochs            Number of epochs to train (default: 1)
--levels            Number of comprehension levels to train for. 2(default), 3 or 6.
--diff              Passage difficulty to use. easy(default), medium, hard or all.
--window            Subsequence window length. Default: 60.
--overlap           Number of frames for overlap of windows. Default: 30.
--filter_type       Filter type to use for data. moving_average(default), kalman or none.
--ma_window         Moving average filter window length. Default: 10.
--gaze_error        Gaze tracking model error to be used for kalman filter. Default: 3.
--stateful          Whether or not to use stateful LSTMs. Default: False.
--weights           Path to weights to be loaded to start training from (optional)
```

The model weights will then be saved to the `models/output` folder.

## Testing

You can test your model using:

```shell
python test.py --model <MODEL_TO_USE> --weights <PATH_TO_WEIGHTS>

Options:
--model             Model to use (lstm(default), gru, cnn_lstm, convlstm2d)
--weights           Path to weights to evaluate model
--levels            Number of comprehension levels to train for. 2(default), 3 or 6.
--diff              Passage difficulty to use. easy(default), medium, hard or all.
--window            Subsequence window length. Default: 60.
--overlap           Number of frames for overlap of windows. Default: 30.
--filter_type       Filter type to use for data. moving_average(default), kalman or none.
--ma_window         Moving average filter window length. Default: 10.
--gaze_error         Gaze tracking model error to be used for kalman filter. Default: 3.
--stateful           Whether or not to use stateful LSTMs. Default: False.
```

## Pre-trained models

Pre-trained models are included in the `models/pretrained` folder. These include the LSTM model weights for 2, 3, 6 comprehension levels and each passage difficulty model.

## Credits
This repository uses suggestions from this exercise on [Human Activity Recognition](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/) to choose and define models.