from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import Normalizer, LabelBinarizer
from tensorflow.keras.utils import to_categorical
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Load train data
train = pd.read_csv('../dataset/binary/trainlabel-bi.csv', header=None, sep=';').iloc[:, 0:1]
trainlabel = pd.read_csv('../dataset/binary/trainlabel-bi.csv', header=None, sep=';').iloc[:, 1:2]

# Load test1 data
test1 = pd.read_csv('../dataset/binary/test1.txt', header=None)
test1label = pd.read_csv('../dataset/binary/test1label.txt', header=None)

# Load test2 data
test2 = pd.read_csv('../dataset/binary/test2.txt', header=None)
test2label = pd.read_csv('../dataset/binary/test2label.txt', header=None)

trainlabel.columns = ['label']
test1label.columns = ['label']
test2label.columns = ['label']

# Combine train and test data for preprocessing
all_data = train._append(test1)._append(test2)
all_labels = trainlabel._append(test1label)._append(test2label)

X = all_data.values.tolist()
X = list(itertools.chain(*X))

valid_chars = {'3': 1, 'c': 2, 'a': 3, '8': 4, 'h': 5, 'g': 6, 'b': 7, 'd': 8, 's': 9, 'r': 10, 'k': 11, 'C': 12, 'P': 13, 'Z': 14, 'm': 15, 'B': 16, 'n': 17, 'i': 18, 'A': 19, 'I': 20, 'v': 21, '4': 22, 'w': 23, '7': 24, '2': 25, 'G': 26, '_': 27, 'e': 28, 'p': 29, ',': 30, 'z': 31, '0': 32, '-': 33, 'E': 34, 'l': 35, '9': 36, 'o': 37, 'u': 38, '5': 39, 'q': 40, 'H': 41, 'f': 42, '1': 43, 'F': 44, 't': 45, 'D': 46, 'y': 47, '.': 48, '6': 49, 'j': 50, 'x': 51}
max_features = 52
maxlen = 214

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
X_padded = sequence.pad_sequences(X1, maxlen=maxlen)

# Convert labels to categorical
y_padded = to_categorical(all_labels)

# Split the combined data back into train, test1, and test2
X_train = X_padded[:len(train)]
y_train = y_padded[:len(train)]

X_test1 = X_padded[len(train):len(train) + len(test1)]
y_test1 = y_padded[len(train):len(train) + len(test1)]

X_test2 = X_padded[len(train) + len(test1):]
y_test2 = y_padded[len(train) + len(test1):]

# Load model
model = load_model('./logs/cnn_lstm/final_cnn_lstm.h5')

def evaluate_model(model, X_test, y_test, filename_prefix):
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='samples')
    recall = recall_score(y_test, y_pred_classes, average='samples')
    f1 = f1_score(y_test, y_pred_classes, average='samples')
    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')

    with open(f'./logs/cnn_lstm/{filename_prefix}_results.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'AUC: {auc_score}\n')

    fpr, tpr, _ = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {filename_prefix}')
    plt.legend(loc="lower right")
    plt.savefig(f'./logs/cnn_lstm/{filename_prefix}_roc_curve.png')
    plt.close()
    
    return accuracy, precision, recall, f1, auc_score

# Evaluate on test1
test1_metrics = evaluate_model(model, X_test1, y_test1, 'test1')

# Evaluate on test2
test2_metrics = evaluate_model(model, X_test2, y_test2, 'test2')

# Plot comparison of test1 and test2
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
test1_scores = test1_metrics
test2_scores = test2_metrics

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, test1_scores, width, label='Test1')
rects2 = ax.bar(x + width/2, test2_scores, width, label='Test2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_title('Comparison of test1 and test2 evaluation metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.savefig('./logs/cnn_lstm/comparison_test1_test2.png')
plt.show()
