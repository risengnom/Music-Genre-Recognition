#!/usr/bin/env python
# coding: utf-8

# In[421]:


import pandas
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

import warnings, os, tensorflow as tf

#env parameters
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)


# In[422]:


num_segments = 19
test_split = 0.3


# In[423]:


data = pandas.read_csv('/home/carsten/workspaces/isws/models/predictions_cnn.csv')
num_test_songs = int(len(data)*test_split)


# In[424]:


genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()


# In[425]:


data = data.drop(['offset', 'duration', 'tempo', 'rms', 'mfcc18', 'mfcc19', 'mfcc20'],axis=1)
data.head()


# In[426]:


for i in range(len(data.filename)):
    res = data.iat[i,0]
    res = res.split('.')[0]
    data.iat[i,0] = res


# In[427]:


genre_list = data.iloc[:, 0]
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(genre_list)


# In[428]:


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, 1:], dtype = float))


# In[429]:


X_test = X[:num_test_songs]
X_train = X[num_test_songs:]
y_test = y[:num_test_songs]
y_train = y[num_test_songs:]


# In[430]:


from keras import models
from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[431]:


val_split = int(19000*0.2)
x_val = X_train[:val_split]
partial_x_train = X_train[val_split:]

y_val = y_train[:val_split]
partial_y_train = y_train[val_split:]
x_norm = preprocessing.scale(x_val)


# In[432]:


model = models.Sequential()
model.add(layers.Dense(35, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(15, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath='./../../../models/ann-model-best.h5', monitor='val_loss', save_best_only=True)]

history = model.fit(partial_x_train,
          partial_y_train,
          epochs=200,
          batch_size=512,
          callbacks=callbacks,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)


# In[433]:


print("Results: " + str(results))


# In[434]:


p = model.predict(X_test)


# In[435]:


from sklearn.metrics import confusion_matrix
print(p.argmax(axis=1))
r = []
for int_res in p.argmax(axis=1):
    r.append(genres[int_res])
cm = confusion_matrix(encoder.inverse_transform(y_test), r, genres)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)


# In[436]:


import pandas as pd, seaborn as sn, matplotlib.pyplot as plt
df_cm = pd.DataFrame(cm, genres,
                  genres)
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size

ax=sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap="BuGn")# font size


# In[437]:


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


# In[438]:


show_summary_stats(history)


# In[439]:


def evaluate_songs(songdata=X_test, true_vals=y_test, model=model):
    pred_genres = []
    prediction = []
    true_genres = []
    predictions = model.predict(songdata)
    for i in range(len(predictions)):
        if i%19 == 0:
            true_genres.append(true_vals[i])
            pred_genres.append(prediction)
            prediction = predictions[i]        
        else:
            prediction = (prediction + predictions[i])/2
    pred_genres.append(prediction)
    np.delete(pred_genres, [0])
    return pred_genres, true_genres


# In[440]:


len(X_test)
pred_genres, true_genres = evaluate_songs()
print(len(true_genres))
c = 0
for i in range(len(pred_genres)):
    if i != 0:
        if pred_genres[i].argmax(axis=0) == true_genres[i-1]:
            c = c + 1
print(c)
acc = c/len(true_genres)*100
print("Accuracy: " + str(acc) + "%")

