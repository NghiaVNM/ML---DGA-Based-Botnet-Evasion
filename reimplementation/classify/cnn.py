from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.datasets import imdb
from sklearn.preprocessing import Normalizer
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.models import load_model
import warnings

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
# valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
valid_chars = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}

# max_features = len(valid_chars) + 1
max_features = 40

# maxlen = np.max([len(x) for x in X])
maxlen = 91

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
num_classes = 21

# Initialize the model
model = Sequential()

# Add Embedding layer
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))

# Add Convolutional layer
model.add(Conv1D(nb_filter, filter_length, padding='valid', activation='relu'))

# Add MaxPooling layer
model.add(MaxPooling1D())

# Add Flatten layer to convert from 3D tensor to 1D vector
model.add(Flatten())

# Add Dense (fully connected) layer
model.add(Dense(hidden_dims, activation='relu'))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define callbacks for model
csv_logger = CSVLogger('./logs2/cnn/training.log')
custom_checkpoint = CustomModelCheckpoint('./logs2/cnn/checkpoint-{epoch:02d}.h5', monitor='val_loss', mode='min', save_best_only=False)
final_checkpoint = CustomModelCheckpoint('./logs2/cnn/final_cnn.h5', monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

# Train the model
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
plt.savefig('./logs2/cnn/training_accuracy.png')
plt.show()