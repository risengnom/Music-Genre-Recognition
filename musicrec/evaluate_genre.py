#!/usr/bin/python

import sys, getopt
import subprocess

import os
import tensorflow as tf
from keras.models import load_model
import pathlib
from pathlib import Path
import librosa
import numpy as np
import progress

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

cnn_modelpath = "/home/carsten/workspaces/isws/models/cnn_gtzan_model_65pct.h5"
duration = 2.97 #Duration of checked segments
n_mels = 64
fft_points = 512
hop_size = int(fft_points/ 2)
fmax = 1500

def download_searched_song(query="ytsearch:bohemian rapsody", output="song.m4a"):
    res = subprocess.check_output(["youtube-dl", "-f", "140", "-o", output, str(query)])
    for line in res.splitlines():
        print(line)

def convert_song(input="song.m4a", output="song.wav"):
    res = subprocess.check_output(["ffmpeg", "-i", input, "-acodec", "pcm_u8", "-ar", "22050", output, "-y", "-loglevel", "panic"])
    for line in res.splitlines():
        print(line)

def remove_songs(input1="song.wav", input2="song.m4a"):
    res = subprocess.check_output(["rm", "song.wav", "song.m4a"])
    for line in res.splitlines():
        print(line)

def calc_features(cnn_model, song="song.wav"):
    progress.startProgress("[INFO]: Calculating features and CVN prediction for " + str(song) + ".")
    chroma_stft = []
    spec_cent = []
    spec_bw = []
    rolloff = []
    zcr = []
    mfcc = []
    cnn_predictions = []
    start = 0
    scan_length = 30
    for i in range(int(scan_length/(duration/2))-1):
        y, sr = librosa.load(song, mono=True, offset=start, duration=duration)
        chroma_stft.append(librosa.feature.chroma_stft(y=y, sr=sr))
        spec_cent.append(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw.append(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff.append(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr.append(librosa.feature.zero_crossing_rate(y))
        mfcc.append(librosa.feature.mfcc(y=y, sr=sr))
        m_sp = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_points,
                                                hop_length=hop_size, n_mels=n_mels,
                                                fmax=fmax)
        m_sp = np.expand_dims(m_sp, 0)
        m_sp = np.expand_dims(m_sp, 3)
        cnn_predictions.append(cnn_model.predict(m_sp))
        start = start + duration/2
        progress.progress((i/scan_length)*100)
    progress.endProgress()
    print("[INFO]: Done.")
    print(cnn_predictions)

def print_instructions():
    print("evaluate-genre.py -s <searchterm> -c <modelpath>")
    print("evaluate-genre.py -u <URL>")

def evaluate(query):
    print("Looking for song: " + query)
    download_searched_song(query)
    print("Converting song.")
    convert_song()
    print("Loading model.")
    cnn_model = load_model(cnn_modelpath)
    print("Calculating features and CNN prediction.")
    calc_features(cnn_model=cnn_model)

    remove_songs()


def main(argv):
    query = ""
    try:
        opts, args = getopt.getopt(argv,"hs:u:m:",["search=","url=","cnnmodel="])
    except getopt.GetoptError:
        print_instructions()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print_instructions()
            sys.exit()
        elif opt in ("-s", "--search"):
            query = "ytsearch:" + str(arg)
            evaluate(query)
        elif opt in ("-u", "--url"):
            query = str(arg)
            evaluate(query)
        elif opt in ("-c", "--cnnmodel"):
            cnn_modelpath = str(arg)

if __name__ == "__main__":
   main(sys.argv[1:])

