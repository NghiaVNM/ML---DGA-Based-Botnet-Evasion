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
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
train = train.iloc[:,0:1]

trainlabel = pd.read_csv('../dataset/classify/trainlabel-multi.csv', header=None)
trainlabel = trainlabel.iloc[:,1:2]

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
# model.load_weights('./logs/cnn/final_cnn.h5')
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model without any callbacks
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Prepare data for new domain
new_domain = "api.watchstadium.com" # 0
encoded_new_domain = [[valid_chars[y] for y in new_domain]]
padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

# Predict classification of new domain
predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)
print("Predicted class for", new_domain, ":", predicted_class)

# Prepare data for new domain
new_domain = "jaunhbjoxi.com" # 4
encoded_new_domain = [[valid_chars[y] for y in new_domain]]
padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

# Predict classification of new domain
predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)
print("Predicted class for", new_domain, ":", predicted_class)

# Prepare data for new domain
new_domain = "elelg850k4qdgf5x7lgvoxy.ddns.net" #2
encoded_new_domain = [[valid_chars[y] for y in new_domain]]
padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

# Predict classification of new domain
predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)
print("Predicted class for", new_domain, ":", predicted_class)
