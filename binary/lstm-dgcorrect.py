from __future__ import print_function
# from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)
from keras.preprocessing import sequence
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
# from keras.utils.np_utils import to_categorical
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

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Train
train = pd.read_csv('../dataset/dgcorrect/binary-train.txt', header=None)

trainlabels = pd.read_csv('../dataset/dgcorrect/binary-label.txt', header=None)
trainlabel = trainlabels.iloc[:,0:1]

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

y_train = np.array(trainlabel)
y_test = np.array(testlabel)
y_test1 = np.array(testlabel1)

embedding_vecor_length = 128
kernel_size = 3 

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/lstm/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('logs/lstm/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, epochs=100,shuffle=True,callbacks=[checkpointer,csv_logger])
model.save("logs/lstm/coomplemodel.hdf5")
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)