#!/usr/bin/env python
# coding: utf-8

# In[2]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
import pathlib
from pathlib import Path
import matplotlib.pyplot as plot
import librosa
    
# import the necessary packages
from musicrec.vgg import VGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from datetime import datetime
import pickle
import cv2
import os
import sys

from keras.models import Sequential


# In[ ]:





# In[3]:


np.set_printoptions(threshold=sys.maxsize)

data_folder = Path("../../../audio/testfiles/GTZAN/genres/")
output_folder = Path("./output/cvnn.model")
spectogram_folder = Path("./img_data/")
# Duration of songsnippet in seconds
duration = 10
# Matplotlib colormap for spectogram
spectogram_cmap = 'binary' 
# Predefined list of genres
pred_genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split() 
#imagePaths = sorted(list(spectogram_folder.list_images(args["dataset"])))
#imagePaths

imagesize_x = 128
imagesize_y = 128


# In[4]:


#Get directories of all songs
songs = []
genres = []

spectograms = []

for g in data_folder.iterdir():
    genres.append(g.name)
    for i in g.iterdir():
        songs.append(i)


# In[5]:


# Calculate all spectograms
cmap = plot.get_cmap(spectogram_cmap)
plot.figure(figsize=(10,10))
spectograms = []

# Iterate through all songs and generate their spactograms. Save them all as images.
for genre in genres:
    pathlib.Path(f'img_data/{genre}').mkdir(parents=True, exist_ok=True)     
for song in songs:
    y, sr = librosa.load(song, mono=True, duration=duration)
    plot.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
    plot.axis('off');
    spec_path = pathlib.Path(f'img_data/{song.parts[-2]}/{".".join(str(song.parts[-1]).split(".")[:2])}.png')
    spectograms.append(spec_path)
    plot.savefig(spec_path)
    plot.clf()


# In[ ]:


# Seed for RNG
random.seed(datetime.now())


# In[ ]:


# Shuffle the spectograms
random.shuffle(spectograms)
spectograms


# In[ ]:


# Import images, convert them to grayscale (one uint8 per pixel) and load them into an array
images = []
for spec_path in spectograms:
    image = cv2.imread(str(spec_path))
    col_pixels = np.array(np.where(image != 255))
    first_col_pixel = col_pixels[:,0]
    last_col_pixel = col_pixels[:,-1]
    image = image[first_col_pixel[0]:last_col_pixel[0], first_col_pixel[1]:last_col_pixel[1]]
    image = cv2.resize(image, (imagesize_x, imagesize_y))
    image_name = str(spec_path) + "-crop.png"
    cv2.imwrite(image_name, image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
images = np.array(images, dtype="float") / 255.0 
data = images


# In[ ]:


# Create array with correct labels for each spectogram
labels = []
for spec in spectograms:
    labels.append(spec.parts[-2])  
labels


# In[ ]:


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
 
# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
np.size(trainY)


# In[ ]:


lb.classes_


# In[ ]:


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
                         height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2,horizontal_flip=True, 
                         fill_mode="nearest")
 
# initialize our VGG-like Convolutional Neural Network
model = VGGNet.build(width=imagesize_x, height=imagesize_y, depth=3, 
                          classes=len(lb.classes_))


# In[ ]:


# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32
 
# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), 
                        validation_data=(testX, testY), 
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)


# In[ ]:


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), 
                            target_names=lb.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
 
# save the model and label binarizer to disk
#print("[INFO] serializing network and label binarizer...")
#model.save(args["model"])
#f = open(args["label_bin"], "wb")
#f.write(pickle.dumps(lb))
#f.close()


# In[ ]:




