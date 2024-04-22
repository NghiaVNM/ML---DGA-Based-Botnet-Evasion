from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
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
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# train
train = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
train = train.iloc[:,0:1]

trainlabel = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
trainlabel = trainlabel.iloc[:,1:2]

X = train.values.tolist()
X = list(itertools.chain(*X))

# test 1
test1 = pd.read_csv('../dataset/classify/test1.txt', header=None)

test1label = pd.read_csv('../dataset/classify/test1label.txt', header=None)

X_test1 = test1.values.tolist()
X_test1 = list(itertools.chain(*X_test1))

# test 2
test2 = pd.read_csv('../dataset/classify/test2.txt', header=None)

test2label = pd.read_csv('../dataset/classify/test2label.txt', header=None)

X_test2 = test2.values.tolist()
X_test2 = list(itertools.chain(*X_test2))

# Generate a dictionary of valid characters
valid_chars = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}
max_features = 40
maxlen = 91

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X1, maxlen=maxlen)

X1_test1 = [[valid_chars[y] for y in x] for x in X_test1]
X_test1 = sequence.pad_sequences(X1_test1, maxlen=maxlen)

X1_test2 = [[valid_chars[y] for y in x] for x in X_test2]
X_test2 = sequence.pad_sequences(X1_test2, maxlen=maxlen)

y_train = np.array(trainlabel)
y_train = to_categorical(y_train)

y_test1 = np.array(test1label)
y_test1 = to_categorical(y_test1)

y_test2 = np.array(test2label)
y_test2 = to_categorical(y_test2)

embedding_vecor_length = 128
num_classes = 21

model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(GRU(128))
model.add(Dropout(0.1))
model.add(Dense(21))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.load_weights('./logs/gru/final_cnn.h5')

model.fit(X_train, y_train, epochs=1, batch_size=32)

def plot_roc_curve(y_true, y_pred, filename):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)

# Predict labels on the test set
y_pred1 = model.predict(X_test1)
y_pred1_classes = np.argmax(y_pred1, axis=1)

# Evaluate the predictions
accuracy = accuracy_score(np.argmax(y_test1, axis=1), y_pred1_classes)
precision = precision_score(np.argmax(y_test1, axis=1), y_pred1_classes, average='weighted')
recall = recall_score(np.argmax(y_test1, axis=1), y_pred1_classes, average='weighted')
f1 = f1_score(np.argmax(y_test1, axis=1), y_pred1_classes, average='weighted')
auc_score = roc_auc_score(y_test1, y_pred1, multi_class='ovr')

# Save the results to a .txt file
with open('./logs/gru/test1_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'AUC: {auc_score}\n')

# Plot and save the ROC curve
plot_roc_curve(np.argmax(y_test1, axis=1), y_pred1_classes, './logs/gru/test1_roc_curve.png')

# Predict labels on the test2 set
y_pred2 = model.predict(X_test2)
y_pred2_classes = np.argmax(y_pred2, axis=1)

# Evaluate the predictions
accuracy = accuracy_score(np.argmax(y_test2, axis=1), y_pred2_classes)
precision = precision_score(np.argmax(y_test2, axis=1), y_pred2_classes, average='weighted')
recall = recall_score(np.argmax(y_test2, axis=1), y_pred2_classes, average='weighted')
f1 = f1_score(np.argmax(y_test2, axis=1), y_pred2_classes, average='weighted')
auc_score = roc_auc_score(y_test1, y_pred1, multi_class='ovr')

# Save the results to a .txt file
with open('./logs/gru/test2_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'AUC: {auc_score}\n')

# Plot and save the ROC curve
plot_roc_curve(np.argmax(y_test2, axis=1), y_pred2_classes, './logs/gru/test2_roc_curve.png')