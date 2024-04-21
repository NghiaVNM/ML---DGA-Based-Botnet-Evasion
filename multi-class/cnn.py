from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337) 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks
from keras.layers import Convolution1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# train
trains = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
train = trains.iloc[:,0:1]
print(train)

trainlabels = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
trainlabel = trainlabels.iloc[:,1:2]
print(trainlabel)

# test 1
test = pd.read_csv('../dataset/dgcorrect-multi/test1.txt', header=None)
print(test)
testlabel = pd.read_csv('../dataset/dgcorrect-multi/test1-label.txt', header=None)
print(testlabel)

# test 2
test1 = pd.read_csv('../dataset/dgcorrect-multi/testing-2.txt', header=None)
print(test1)
testlabel1 = pd.read_csv('../dataset/dgcorrect-multi/testing-2-label.txt', header=None)
print(testlabel1)

X = train.values.tolist()
X = list(itertools.chain(*X))

T = test.values.tolist()
T = list(itertools.chain(*T))

T1 = test1.values.tolist()
T1 = list(itertools.chain(*T1))

# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])
print(maxlen)


# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]

T11 = [[valid_chars[y] for y in x] for x in T]

T12 = [[valid_chars[y] for y in x] for x in T1]


X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X_test = sequence.pad_sequences(T11, maxlen=maxlen)

X_test1 = sequence.pad_sequences(T12, maxlen=maxlen)

y_trainn = np.array(trainlabel)
y_testn = np.array(testlabel)
y_test1n = np.array(testlabel1)

y_train= to_categorical(y_trainn)
y_test= to_categorical(y_testn)
y_test1= to_categorical(y_test1n)

hidden_dims=128
nb_filter = 32
filter_length =3 
embedding_vecor_length = 128

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='valid',
                        activation='relu',
                        strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(21))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('logs/cnn/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, epochs=100,shuffle=True,callbacks=[checkpointer,csv_logger])
model.save("logs/cnn/coomplemodel.hdf5")
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)


