from flask import Flask, request, jsonify
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import model_from_json
import numpy as np

app = Flask(__name__)

# Classification dictionary
valid_chars_classify = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}
max_features_classify = 40
maxlen_classify = 91

# Convert domain to sequence of integers
def convert_domain(domain):
    encoded_new_domain = [[valid_chars_classify[y] for y in domain]]
    padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen_classify)
    return padded_new_domain

def get_name_classify(index):
    names = [
        'lành tính',
        'banjori',
        'corebot',
        'dircrypt',
        'dnschanger',
        'fobber',
        'murofet',
        'necurs',
        'newgoz',
        'padcrypt',
        'proslikefan',
        'pykspa',
        'qadars',
        'qakbot',
        'ramdo',
        'ranbyus',
        'simda',
        'suppobox',
        'symmi',
        'tempedreve',
        'tinba'
    ]

    if 0 <= index < len(names):
        return names[index]
    else:
        return 'Error: Unknown class'

# Classification model

# Load classification model
model_cnn_classify = load_model('./classify/logs/cnn/final_cnn.h5')
model_rnn_classify = load_model('./classify/logs/rnn/final_rnn.h5')
model_gru_classify = load_model('./classify/logs/gru/final_gru.h5')
model_lstm_classify = load_model('./classify/logs/lstm/final_lstm.h5')
model_cnn_lstm_classify = load_model('./classify/logs/cnn_lstm/final_cnn_lstm.h5')

# Classification function

# CNN
def cnn_classify (domain):
    domain = convert_domain(domain)

    # Predict classification of new domain
    predicted_class = np.argmax(model_cnn_classify.predict(domain), axis=-1)
    return predicted_class



@app.route('/api', methods=['GET'])
def get_data():
    # This is where you would retrieve and return some data. 
    # For this example, we'll just return a simple message.
    return jsonify({'message': 'Hello, World!'})

@app.route('/api', methods=['POST'])
def post_data():
    data = request.get_json()

    type = data.get('type')
    model = data.get('model')
    domain = data.get('domain')

    if(type == 'classify'):
        if(model == 'cnn'):
            result = cnn_classify(domain)
            # Convert numpy.ndarray to list
            result = result.tolist()
            result = get_name_classify(result[0])
            return jsonify({'result': result}), 201

if __name__ == '__main__':
    app.run(debug=True, port=5001)