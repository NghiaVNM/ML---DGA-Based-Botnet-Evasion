from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import Normalizer
from tensorflow.keras.utils import to_categorical
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from sklearn.model_selection import train_test_split

# train
train = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
train = train.iloc[:,0:1]

trainlabel = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
trainlabel = trainlabel.iloc[:,1:2]

# test 1
test1 = pd.read_csv('../dataset/classify/test1.txt', header=None)
test1label = pd.read_csv('../dataset/classify/test1label.txt', header=None)

# test 2
test2 = pd.read_csv('../dataset/classify/test2.txt', header=None)
test2label = pd.read_csv('../dataset/classify/test2label.txt', header=None)

train = train._append(test1)
train = train._append(test2)

trainlabel.columns = ['label']
test1label.columns = ['label']
test2label.columns = ['label']

trainlabel = trainlabel._append(test1label)
trainlabel = trainlabel._append(test2label)

X = train.values.tolist()
X = list(itertools.chain(*X))

# Generate a dictionary of valid characters
valid_chars = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}
max_features = 40
maxlen = 91

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X1, maxlen=maxlen)

y_train = np.array(trainlabel)
y_train = to_categorical(y_train)

# Model parameters
num_classes = 21

model = load_model('./logs/lstm/final_lstm.h5')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

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
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate the predictions
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)
precision = precision_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr')

# Save the results to a .txt file
with open('./logs/lstm/test_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'AUC: {auc_score}\n')

# Plot and save the ROC curve
plot_roc_curve(np.argmax(y_test, axis=1), y_pred_classes, './logs/lstm/test_roc_curve.png')