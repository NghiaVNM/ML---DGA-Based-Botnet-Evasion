from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from tensorflow.keras.utils import to_categorical
import itertools

# Generate a dictionary of valid characters
valid_chars = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}

max_features = 40
maxlen = 91

model = load_model('./logs/cnn/checkpoint-00.h5')

while True:
    # Prepare data for new domain
    new_domain = input("Enter a domain name: ")
    encoded_new_domain = [[valid_chars[y] for y in new_domain]]
    padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

    # Predict classification of new domain
    predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)
    print("Predicted class for", new_domain, ":", predicted_class)
