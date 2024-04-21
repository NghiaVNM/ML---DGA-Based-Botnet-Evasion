from __future__ import print_function
# from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
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
from keras.layers import Convolution1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)

# train
trains = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
train = trains.iloc[:,0:1]
# print(train)

trainlabels = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
trainlabel = trainlabels.iloc[:,1:2]
# print(trainlabel)

# test 1
test = pd.read_csv('../dataset/dgcorrect-multi/test1.txt', header=None)
# print(test)

testlabel = pd.read_csv('../dataset/dgcorrect-multi/test1-label.txt', header=None)
# print(testlabel)

# test 2
test1 = pd.read_csv('../dataset/dgcorrect-multi/testing-2.txt', header=None)
# print(test1)
testlabel1 = pd.read_csv('../dataset/dgcorrect-multi/testing-2-label.txt', header=None)
# print(testlabel1)

X = train.values.tolist()
X = list(itertools.chain(*X))
# print (X)

T = test.values.tolist()
T = list(itertools.chain(*T))
# print (T)

T1 = test1.values.tolist()
T1 = list(itertools.chain(*T1))
# print (T1)

# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
print(valid_chars)

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])
print(maxlen)


# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
# print(X1)
T11 = [[valid_chars[y] for y in x] for x in T]
# print(T11)
T12 = [[valid_chars[y] for y in x] for x in T1]
# print(T12)

X_train = sequence.pad_sequences(X1, maxlen=maxlen)
# print(X_train)

X_test = sequence.pad_sequences(T11, maxlen=maxlen)
# print(X_test)

X_test1 = sequence.pad_sequences(T12, maxlen=maxlen)
# print(X_test1)

y_trainn = np.array(trainlabel)
# print(y_trainn)

y_testn = np.array(testlabel)
# print(y_testn)

y_test1n = np.array(testlabel1)
# print(y_test1n)

y_train= to_categorical(y_trainn)
print(y_train[:5])

y_test= to_categorical(y_testn)
print(y_test[:5])

y_test1= to_categorical(y_test1n)
print(y_test1[:5])

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

'''
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('logs/cnn/training_set_lstmanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=32, nb_epoch=1000,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
score, acc = model.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
'''

# try using different optimizers and different optimizer configs
model.load_weights("logs/cnn/checkpoint-54.hdf5")

# def preprocess_url(url, maxlen, valid_chars):
#     url_int = [valid_chars[y] for y in url]
#     url_int_pad = sequence.pad_sequences([url_int], maxlen=maxlen)
#     return url_int_pad

# url_to_check1 = "m8aoicy8qous.top"
# url_to_check2 = "fiddlrxsat.com"

# url_to_check_int1 = preprocess_url(url_to_check1, maxlen, valid_chars)
# url_to_check_int2 = preprocess_url(url_to_check2, maxlen, valid_chars)

# prediction1 = (model.predict(url_to_check_int1) > 0.5).astype("int32")
# prediction2 = (model.predict(url_to_check_int2) > 0.5).astype("int32")
# print(prediction1)
# print(prediction2)
# if prediction1 == 1:
#     print("The URL '{}' is negative (benign).".format(url_to_check1))
# else:
#     print("The URL '{}' is positive (malicious).".format(url_to_check1))
    

# if prediction2 == 1:
#     print("The URL '{}' is negative (benign).".format(url_to_check2))
# else:
#     print("The URL '{}' is positive (malicious).".format(url_to_check2))

# y_pred = model.predict(X_test)
# for i in range(len(X_test)):
#     print(f"Sample {i+1}:")
#     print(f"Expected: {np.argmax(y_testn[i])}")
#     print(f"Predicted: {np.argmax(y_pred[i])}")
#     print("-------------------------")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test1n, y_pred)
recall = recall_score(y_test1n, y_pred , average="weighted")
precision = precision_score(y_test1n, y_pred , average="weighted")
f1 = f1_score(y_test1n, y_pred, average="weighted")

print("confusion matrix")
print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)

np.savetxt("multi-d2.txt",y_pred, fmt="%01d")
'''
cm = metrics.confusion_matrix(y_test, y_pred)
print("==============================================")
print(cm)
tp = cm[0][0]
fp = cm[0][1]
tn = cm[1][1]
fn = cm[1][0]
print("tp")
print(tp)
print("fp")
print(fp)
print("tn")
print(tn)
print("fn")
print(fn)

print("LSTM acc")
Acc = float(tp+tn)/float(tp+fp+tn+fn)
print(Acc)
print("precision")
prec = float(tp)/float(tp+fp)
print(prec)
print("recall")
rec = float(tp)/float(tp+fn)
print(rec)
print("F-score")
fs = float(2*tp)/float((2*tp)+fp+fn)
print(fs)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
'''


