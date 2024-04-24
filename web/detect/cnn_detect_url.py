from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from tensorflow.keras.utils import to_categorical
import numpy as np

valid_chars = {'3': 1, 'c': 2, 'a': 3, '8': 4, 'h': 5, 'g': 6, 'b': 7, 'd': 8, 's': 9, 'r': 10, 'k': 11, 'C': 12, 'P': 13, 'Z': 14, 'm': 15, 'B': 16, 'n': 17, 'i': 18, 'A': 19, 'I': 20, 'v': 21, '4': 22, 'w': 23, '7': 24, '2': 25, 'G': 26, '_': 27, 'e': 28, 'p': 29, ',': 30, 'z': 31, '0': 32, '-': 33, 'E': 34, 'l': 35, '9': 36, 'o': 37, 'u': 38, '5': 39, 'q': 40, 'H': 41, 'f': 42, '1': 43, 'F': 44, 't': 45, 'D': 46, 'y': 47, '.': 48, '6': 49, 'j': 50, 'x': 51}
max_features = 52
maxlen = 214

model = load_model('./logs/cnn/final_cnn.h5')

while True:
    # Prepare data for new domain
    new_domain = input("Enter a domain name: ")
    encoded_new_domain = [[valid_chars[y] for y in new_domain]]
    padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

    # Predict classification of new domain
    predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)
    print("Predicted class for", new_domain, ":", predicted_class)
