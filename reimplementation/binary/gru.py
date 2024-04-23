from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, GRU
from sklearn.preprocessing import Normalizer
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = None
        
        if self.mode == 'min':
            self.best_value = np.Inf
        else:
            self.best_value = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            warnings.warn("CustomModelCheckpoint requires %s available!" % self.monitor, RuntimeWarning)
            return

        if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            if self.save_best_only:
                filepath = self.filepath.format(epoch=epoch, **logs)
                self.best_value = current_value
                self.model.save(filepath)
            else:
                self.best_value = current_value
                filepath = self.filepath.format(epoch=epoch, **logs)
                self.model.save(filepath)
        else:
            pass

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

# Generate a dictionary of valid characters
valid_chars = {'3': 1, 'c': 2, 'a': 3, '8': 4, 'h': 5, 'g': 6, 'b': 7, 'd': 8, 's': 9, 'r': 10, 'k': 11, 'C': 12, 'P': 13, 'Z': 14, 'm': 15, 'B': 16, 'n': 17, 'i': 18, 'A': 19, 'I': 20, 'v': 21, '4': 22, 'w': 23, '7': 24, '2': 25, 'G': 26, '_': 27, 'e': 28, 'p': 29, ',': 30, 'z': 31, '0': 32, '-': 33, 'E': 34, 'l': 35, '9': 36, 'o': 37, 'u': 38, '5': 39, 'q': 40, 'H': 41, 'f': 42, '1': 43, 'F': 44, 't': 45, 'D': 46, 'y': 47, '.': 48, '6': 49, 'j': 50, 'x': 51}

max_features = 52
maxlen = 214

X1 = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X1, maxlen=maxlen)

y_train = np.array(trainlabel)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(GRU(128))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define callbacks for model
csv_logger = CSVLogger('./logs/gru/training.log')
custom_checkpoint = CustomModelCheckpoint('./logs/gru/checkpoint-{epoch:02d}.h5', monitor='val_loss', mode='min', save_best_only=False)
final_checkpoint = CustomModelCheckpoint('./logs/gru/final_gru.h5', monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=10, batch_size=32, 
                    callbacks=[csv_logger, custom_checkpoint, final_checkpoint, early_stopping])

# Plot training accuracy
train_acc = history.history['accuracy']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./logs/gru/training_accuracy.png')
plt.show()
