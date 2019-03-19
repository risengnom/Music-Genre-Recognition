#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
get_ipython().run_line_magic('matplotlib', 'inline')

import pathlib
from pathlib import Path
import matplotlib.pyplot as plot
import librosa
import tensorflow as tf
import keras
from keras import regularizers
from keras.models import load_model, Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D,                          Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
    
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from numpy import argmax
from imutils import paths
import numpy as np

from datetime import datetime
import pickle, random, os, sys, json, re, getopt, warnings

#Visualization
import seaborn as sn
import pandas as pd

#env parameters
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)


# In[2]:


#Parameters
#Folders
data_folder = Path("../../../audio/testfiles/GTZAN/genres/")
output_root = './../../../models/'
output_folder = Path('./output/cvnn_16-3.model')
output_model = output_root + '/cnn_dong_model_weights.h5'
output_architecture = output_root + '/cnn_dong_model_architecture.json'
output_whole = output_root + 'cnn_dong_model_whole.h5'
output_best_model = output_root + 'best_model.h5'
output_label = output_root + 'label.pkl'
output_test_paths = output_root + 'test_paths.pkl'

# Duration of songsnippet in seconds
duration = 2.97
start_offset = 0
epochs = 400
num_segments = 19

batch_size = 64

es_patience = 20
return_train_and_test = 1

sr = 22050 # Sampling rate

#Parameters for mel spec
fmax = 1500 # maximum frequency considered
fft_points = 512
fft_dur = fft_points * 1.0 / sr # 23ms windows
hop_size = int(fft_points/ 2) # 50% overlap between consecutive frames
n_mels = 64

#Segment duration
num_fft_windows = 256 #per Segment
segment_in_points = num_fft_windows * 255
segment_dur = segment_in_points * 1.0 / sr

input_shape=(64, 256, 1)

randomseed = 11
#randomseed = datetime.now()
# Seed for RNG
random.seed(randomseed)


# In[3]:


#Get directories of all songs
songs = []
genres = []

for g in data_folder.iterdir():
    genres.append(g.name)
    for i in g.iterdir():
        songs.append(i)


# In[4]:


#shuffle songs to keep segements together
random.shuffle(songs)
print("[Info]: Loaded and shuffled " + str(len(songs)) + " songs from " + str(len(genres)) + " genres.")


# In[5]:


#loading with different numbers of segements for data-augmentation
#!Initialize "spectograms" before running
def load_specs(data = songs, num_segments = 1):
    spectograms = []
    for song in data:
        offset = start_offset
        for i in range(num_segments):
            y, sr = librosa.load(song, mono=True, offset=offset, duration=duration)
            m_sp = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_points,
                                                  hop_length=hop_size, n_mels=n_mels,
                                                  fmax=fmax)
            label = song.parts[-2]
            spectograms.append( (m_sp, label, song, offset, duration) )
            offset = offset + duration/2
        input_shape = m_sp.shape + (1,)
    return spectograms


# In[6]:


print("[Info]: Generating Spectograms for " + str(num_segments) + " segments per song.")
spectograms = load_specs(data = songs, num_segments = num_segments)
labels = []
for i in spectograms:
    labels.append(i[1]) 


# In[7]:


print("[Info]: Total number of samples: ", len(input_shape))

#Set dynamic input shape
input_shape = np.shape(spectograms[0][0]) + (1,)
print("Input shape: " + str(input_shape))


# In[8]:


print("[Info]: Splitting into train and testing data.")
#Split into Train and Testing
testsplit = len(spectograms)*0.7    #70% train-test-split
train = spectograms[:int(testsplit)]
test = spectograms[int(testsplit):]

x_train, y_train, p_train, offset_train, duration_train = zip(*train)
x_test, y_test, p_test, offset_test, duration_test = zip(*test)

#Fit Dimensions
x_train = np.array([x.reshape(input_shape) for x in x_train])
x_test = np.array([x.reshape(input_shape) for x in x_test])


# In[9]:


#Binarize Labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


# In[10]:


def dump_testspecs_labels():
    with open(output_label, 'wb') as f:
        pickle.dump(lb, f)

    if return_train_and_test == 1:
        r_paths = p_test + (p_train)
        r_offsets = offset_test + (offset_train)
        r_durations = duration_test + (duration_train)
    else:
        r_paths = p_test
        r_offsets = offset_test
        r_durations = duration_test

    r_values = [r_paths, r_offsets, r_durations]

    with open(output_test_paths, 'wb') as f:
        pickle.dump(r_values, f)


# In[11]:


print("[Info]: Dumping data for prediction.")
dump_testspecs_labels()


# In[12]:


#model for GTZAN
def cnn_gtzan_model_build():

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.3))
    model.add(Dense(len(genres), activation='softmax'))
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    print(model.summary)
    return model

def train_model():
    model = cnn_gtzan_model_build()

    callbacks = [EarlyStopping(monitor='val_loss', patience=es_patience),
                 ModelCheckpoint(filepath=output_best_model, monitor='val_loss', save_best_only=True)]

    H = model.fit(
        x=x_train, 
        y=y_train,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size,
        validation_data= (x_test, y_test))

    score = model.evaluate(x=x_test,y=y_test)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return (model, H)

def save_model():
    # Save the weights
    model.save_weights(output_model)

    # Save the model architecture
    with open(output_architecture, 'w') as f:
        f.write(model.to_json())

    # Save complete model
    model.save(output_whole)


# In[13]:


print("[Info]: Training model.")
model, history = train_model()


# In[14]:


print("[Info]: Saving model.")
save_model()


# In[15]:


def evaluate_song(song, model=model, duration=duration):
    start = 0
    r = 0
    for i in range(int(30/(duration/2))-1):
        y, sr = librosa.load(song, mono=True, offset=start, duration=duration)
        m_sp = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_points,
                                              hop_length=hop_size, n_mels=n_mels,
                                              fmax=fmax)
        m_sp = np.expand_dims(m_sp, 0)
        m_sp = np.expand_dims(m_sp, 3)
        if r == 0:
            prediction = model.predict(m_sp)
            r = 1
        else:
            prediction = (prediction + model.predict(m_sp))/2
        start = start + duration/2
    return prediction
        
def evaluate_batch(batch):
    predictions = []
    for song in batch:
        predictions.append(evaluate_song(song))
    return predictions

def compare_batch(predictions):
    i = 0
    l = lb.inverse_transform(y_test)
    counter = 0
    for prediction in predictions:
        tr_value = str(l[i])
        pr_value = re.sub('[\[\]\']', '', str(lb.classes_[prediction.argmax(axis=1)]))
        #print(tr_value + "=>" + pr_value)
        if tr_value == pr_value:
            counter += 1
        i += 1
    acc = counter/len(predictions)*100
    print("Acc: " + str(acc) + "%")


# In[16]:


model = load_model(output_whole)


# In[17]:


predictions = evaluate_batch(p_test)
compare_batch(predictions)


# In[18]:


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x_test, batch_size=batch_size)


# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()


# In[19]:


print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1), 
                            target_names=lb.classes_))


# In[ ]:





# In[20]:


cm = confusion_matrix(lb.inverse_transform(y_test), lb.classes_[predictions.argmax(axis=1)], lb.classes_)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)


# In[21]:


df_cm = pd.DataFrame(cm, lb.classes_,
                  lb.classes_)
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size

ax=sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap="BuGn")# font size


# In[22]:


history


# In[23]:


def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[24]:


show_summary_stats(history)

