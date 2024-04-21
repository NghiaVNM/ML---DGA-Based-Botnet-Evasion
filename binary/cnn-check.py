from __future__ import print_function
# from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
# from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger, ModelCheckpoint
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Train
train = pd.read_csv('../dataset/dgcorrect/trainlabel-binary.csv', header=None, sep=';')
train = train.iloc[:,0:1]

trainlabels = pd.read_csv('../dataset/dgcorrect/trainlabel-binary.csv', header=None, sep=';')
trainlabel = trainlabels.iloc[:,1:2]

# Test 1
test = pd.read_csv('../dataset/dgcorrect/test1.txt', header=None)

testlabels = pd.read_csv('../dataset/dgcorrect/test1label.txt', header=None)
testlabel = testlabels.iloc[:,0:1]

# Test 2
test1 = pd.read_csv('../dataset/dgcorrect/test2.txt', header=None)

testlabels1 = pd.read_csv('../dataset/dgcorrect/test2label.txt', header=None)
testlabel1 = testlabels1.iloc[:,0:1]

X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))

T1 = test1.values.tolist()
T1 = list(itertools.chain(*T1))

all_data = X + T + T1
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(all_data)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]

T11 = [[valid_chars[y] for y in x] for x in T]

T12 = [[valid_chars[y] for y in x] for x in T1]


X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X_test = sequence.pad_sequences(T11, maxlen=maxlen)

X_test1 = sequence.pad_sequences(T12, maxlen=maxlen)

y_train = np.array(trainlabel)
y_test = np.array(testlabel)
y_test1 = np.array(testlabel1)

hidden_dims=128
nb_filter = 32
filter_length =3 
embedding_vecor_length = 128
kernel_size = 3 

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(filters=nb_filter,
                        kernel_size=kernel_size,
                        activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model.load_weights("logs/cnn/coomplemodel.hdf5")
model.load_weights("logs/cnn/checkpoint-29.hdf5")

def preprocess_url(url, maxlen, valid_chars):
    url_int = [valid_chars[y] for y in url]
    url_int_pad = sequence.pad_sequences([url_int], maxlen=maxlen)
    return url_int_pad

url_to_check1 = "google.com" # Benign
url_to_check2 = "c4w6wpg81xsbopy8a67.ddns.net" # Botnet

url_to_check_int1 = preprocess_url(url_to_check1, maxlen, valid_chars)
url_to_check_int2 = preprocess_url(url_to_check2, maxlen, valid_chars)

prediction1 = (model.predict(url_to_check_int1) > 0.5).astype("int32")
prediction2 = (model.predict(url_to_check_int2) > 0.5).astype("int32")
print(prediction1)
print(prediction2)
if prediction1 == 1:
    print("The URL '{}' is positive (malicious).".format(url_to_check1))
else:
    print("The URL '{}' is negative (benign).".format(url_to_check1))
    

if prediction2 == 1:
    print("The URL '{}' is positive (malicious).".format(url_to_check2))
else:
    print("The URL '{}' is negative (benign).".format(url_to_check2))