import pandas as pd
import numpy as np
import itertools
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D

# Train
train = pd.read_csv('../dataset/dgcorrect/trainlabel-binary.csv', header=None, sep=';')
train = train.iloc[:,0:1]
print(train)

trainlabels = pd.read_csv('../dataset/dgcorrect/trainlabel-binary.csv', header=None, sep=';')
trainlabel = trainlabels.iloc[:,1:2]
print(trainlabel)

# Test 1
test = pd.read_csv('../dataset/dgcorrect/test1.txt', header=None)
print(test)

testlabels = pd.read_csv('../dataset/dgcorrect/test1label.txt', header=None)
testlabel = testlabels.iloc[:,0:1]
print(testlabel)

# Test 2
test1 = pd.read_csv('../dataset/dgcorrect/test2.txt', header=None)
print(test1)

testlabels1 = pd.read_csv('../dataset/dgcorrect/test2label.txt', header=None)
testlabel1 = testlabels1.iloc[:,0:1]
print(testlabel1)

X = train.values.tolist()
X = list(itertools.chain(*X))


T = test.values.tolist()
T = list(itertools.chain(*T))

T1 = test1.values.tolist()
T1 = list(itertools.chain(*T1))

# Combine all datasets into one list
all_data = X + T + T1
# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(all_data)))}

max_features = len(valid_chars) + 1

maxlen = np.max([len(x) for x in X])

X1 = [[valid_chars[y] for y in x] for x in X]

X_train = sequence.pad_sequences(X1, maxlen=maxlen)

y_train = np.array(trainlabel)

hidden_dims = 128
nb_filter = 32
embedding_vecor_length = 128
kernel_size = 3

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(Convolution1D(filters=nb_filter, kernel_size=kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("logs/cnn/checkpoint-01.hdf5")

def preprocess_url(url, maxlen, valid_chars):
    url_int = [valid_chars[y] for y in url]
    url_int_pad = sequence.pad_sequences([url_int], maxlen=maxlen)
    return url_int_pad

url_to_check1 = "c4w6wpg81xsbopy8a67.ddns.net"
url_to_check_int1 = preprocess_url(url_to_check1, maxlen, valid_chars)
prediction1 = (model.predict(url_to_check_int1) > 0.5).astype("int32")
if prediction1 == 1:
    print("The URL '{}' is positive (malicious).".format(url_to_check1))
else:
    print("The URL '{}' is negative (benign).".format(url_to_check1))

url_to_check2 = "google.com"
url_to_check_int2 = preprocess_url(url_to_check2, maxlen, valid_chars)
prediction2 = (model.predict(url_to_check_int2) > 0.5).astype("int32")
if prediction2 == 1:
    print("The URL '{}' is positive (malicious).".format(url_to_check2))
else:
    print("The URL '{}' is negative (benign).".format(url_to_check2))
