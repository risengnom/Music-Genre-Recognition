#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from keras.models import load_model
import pathlib
from pathlib import Path
import librosa
import numpy as np
import pickle
import re
import csv
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


# In[2]:


def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


# In[3]:


# parameters
duration = 2.97
sr = 22050 # if sampling rate is different, resample it to this
input_root = './../../../models/'
input_whole = input_root + 'cnn_dong_model_whole.h5'
input_label = input_root + 'label.pkl'
input_test_paths = input_root + 'test_paths.pkl'
default_song = Path("../../../audio/testfiles/GTZAN/genres/rock/rock.00003.wav")
output_test_results = input_root + 'predictions_cnn.pkl'
output_test_results_csv = input_root + 'predictions_cnn.csv'
fmax = 1500 # maximum frequency considered
fft_window_points = 512
fft_window_dur = fft_window_points * 1.0 / sr # 23ms windows
hop_size = int(fft_window_points/ 2) # 50% overlap between consecutive frames
n_mels = 64


# In[4]:


#load model, labels and paths for songs to predict
print("[INFO]: Initializing Prediction. Loading model: " + str(input_whole))
model = load_model(input_whole)
with open(input_label, 'rb') as f:
    lb = pickle.load(f)
print("[INFO]: Done loading model.")

print("[INFO]: Loading song specifications: " + str(input_test_paths))
with open(input_test_paths, 'rb') as f:
    data = pickle.load(f)
print("[INFO]: Done loading song specifications.")


# In[5]:


paths = data[0]
offsets = data [1]
durations = data [2]
if len(paths) == len(offsets) and len(offsets) == len(durations):
    print("[INFO]: Ingested dimensions are fine! Amount of Datapoints: " + str(len(paths)))
else:
    print("[Error]: Dimensions of read file invalid!")
    raise


# In[6]:


#create header
print("[INFO]: Writing .csv to following location: " + str(output_test_results_csv))
header = 'filename offset duration chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += f' blues classical country disco hiphop jazz metal pop reggae rock'
header = header.split()
#write header to .csv
file = open(output_test_results_csv, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)


# In[7]:


startProgress("[INFO]: Calculating and saving features and CVN prediction for " + str(len(paths)) + " samples...")
p_songs = []
i = 0

for path in paths:
    y, sr = librosa.load(path, mono=True, offset=offsets[i], duration=durations[i])
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #rmse = librosa.feature.rmse(y=y, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='reflect')
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    m_sp = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_window_points,
                                              hop_length=hop_size, n_mels=n_mels,
                                              fmax=fmax)
    m_sp = np.expand_dims(m_sp, 0)
    m_sp = np.expand_dims(m_sp, 3)
    cvn_prediction = model.predict(m_sp)
    to_append = f'{path.name} {offsets[i]} {durations[i]} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for e in cvn_prediction:
        e = re.sub('[\[\]]', '', str(e))
        to_append += f' {e}'
    #to_append += f' {str(lb.classes_[prediction.argmax(axis=-1)])}'
    arr = (to_append.split())
    file = open(output_test_results_csv, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(arr)
    p_songs.append(arr)
    i = i+1
    progress((i/len(paths))*100)
endProgress()
print("[INFO]: Done.")


# In[ ]:


with open(output_test_results, 'wb') as f:
    pickle.dump(p_songs, f)

