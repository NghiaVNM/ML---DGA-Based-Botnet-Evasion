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
train = pd.read_csv('../dataset/binary/trainlabel-bi.csv', header=None, sep=';')
train = train.iloc[:,0:1]

trainlabel = pd.read_csv('../dataset/binary/trainlabel-bi.csv', header=None, sep=';')
trainlabel = trainlabel.iloc[:,1:2]

# test 1
test1 = pd.read_csv('../dataset/binary/test1.txt', header=None)
test1label = pd.read_csv('../dataset/binary/test1label.txt', header=None)

# test 2
test2 = pd.read_csv('../dataset/binary/test2.txt', header=None)
test2label = pd.read_csv('../dataset/binary/test2label.txt', header=None)

train = train._append(test1)
train = train._append(test2)

trainlabel.columns = ['label']
test1label.columns = ['label']
test2label.columns = ['label']

trainlabel = trainlabel._append(test1label)
trainlabel = trainlabel._append(test2label)

X = train.values.tolist()
X = list(itertools.chain(*X))

valid_chars = {'3': 1, 'c': 2, 'a': 3, '8': 4, 'h': 5, 'g': 6, 'b': 7, 'd': 8, 's': 9, 'r': 10, 'k': 11, 'C': 12, 'P': 13, 'Z': 14, 'm': 15, 'B': 16, 'n': 17, 'i': 18, 'A': 19, 'I': 20, 'v': 21, '4': 22, 'w': 23, '7': 24, '2': 25, 'G': 26, '_': 27, 'e': 28, 'p': 29, ',': 30, 'z': 31, '0': 32, '-': 33, 'E': 34, 'l': 35, '9': 36, 'o': 37, 'u': 38, '5': 39, 'q': 40, 'H': 41, 'f': 42, '1': 43, 'F': 44, 't': 45, 'D': 46, 'y': 47, '.': 48, '6': 49, 'j': 50, 'x': 51}
max_features = 52
maxlen = 214

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X1, maxlen=maxlen)

y_train = np.array(trainlabel)
y_train = to_categorical(y_train)

# Model parameters
hidden_dims = 128
nb_filter = 32
filter_length = 3
embedding_vecor_length = 128
num_classes = 2

model = load_model('./logs/rnn/final_rnn.h5')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def plot_roc_curve(y_true, y_pred, filename):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
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
y_pred_classes = (y_pred > 0.5).astype(int)  # Threshold can be adjusted based on your needs

# Evaluate the predictions
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='samples')  # Changed to 'samples' for multi-label
recall = recall_score(y_test, y_pred_classes, average='samples')  # Changed to 'samples' for multi-label
f1 = f1_score(y_test, y_pred_classes, average='samples')  # Changed to 'samples' for multi-label
auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')  # Added 'average' parameter

# Save the results to a .txt file
with open('./logs/rnn/test_results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'AUC: {auc_score}\n')

# Plot and save the ROC curve
plot_roc_curve(np.argmax(y_test, axis=1), y_pred_classes[:, 1], './logs/rnn/test_roc_curve.png')