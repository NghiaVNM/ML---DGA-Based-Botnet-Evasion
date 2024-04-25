from flask import Flask, request, jsonify
from keras.preprocessing import sequence
from keras.models import load_model
import numpy as np
from colorama import Fore

# Dictionary
max_features_classify = 40
maxlen_classify = 91
valid_chars_classify = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}

max_features_detect = 52
maxlen_detect = 214
valid_chars_detect = {'3': 1, 'c': 2, 'a': 3, '8': 4, 'h': 5, 'g': 6, 'b': 7, 'd': 8, 's': 9, 'r': 10, 'k': 11, 'C': 12, 'P': 13, 'Z': 14, 'm': 15, 'B': 16, 'n': 17, 'i': 18, 'A': 19, 'I': 20, 'v': 21, '4': 22, 'w': 23, '7': 24, '2': 25, 'G': 26, '_': 27, 'e': 28, 'p': 29, ',': 30, 'z': 31, '0': 32, '-': 33, 'E': 34, 'l': 35, '9': 36, 'o': 37, 'u': 38, '5': 39, 'q': 40, 'H': 41, 'f': 42, '1': 43, 'F': 44, 't': 45, 'D': 46, 'y': 47, '.': 48, '6': 49, 'j': 50, 'x': 51}

# Convert domain to sequence of integers
def convert_domain_clasify(domain):
    encoded_new_domain = [[valid_chars_classify[y] for y in domain]]
    padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen_classify)
    return padded_new_domain

def convert_domain_detect(domain):
    encoded_new_domain = [[valid_chars_detect[y] for y in domain]]
    padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen_detect)
    return padded_new_domain

# Get name of class
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

def get_name_detect(index):
    names = [
        'lành tính',
        'malicious',
    ]

    if 0 <= index < len(names):
        return names[index]
    else:
        return 'Error: Unknown class'

# Load model
model_cnn_classify = load_model('./classify/logs/cnn/final_cnn.h5')
model_rnn_classify = load_model('./classify/logs/rnn/final_rnn.h5')
model_gru_classify = load_model('./classify/logs/gru/final_gru.h5')
model_lstm_classify = load_model('./classify/logs/lstm/final_lstm.h5')
model_cnn_lstm_classify = load_model('./classify/logs/cnn_lstm/final_cnn_lstm.h5')

model_cnn_detect = load_model('./detect/logs/cnn/final_cnn.h5')
model_rnn_detect = load_model('./detect/logs/rnn/final_rnn.h5')
model_gru_detect = load_model('./detect/logs/gru/final_gru.h5')
model_lstm_detect = load_model('./detect/logs/lstm/final_lstm.h5')
model_cnn_lstm_detect = load_model('./detect/logs/cnn_lstm/final_cnn_lstm.h5')

# Function
def predict_url(domain, model, type):
    if type == 'classify':
        domain = convert_domain_clasify(domain)
    elif type == 'detect':
        domain = convert_domain_detect(domain)
    else:
        raise ValueError("Invalid type. Type must be 'classify' or 'detect'.")

    predicted_class = np.argmax(model.predict(domain), axis=-1)
    return predicted_class

# API
cnn_detect = [0.9855432211548814, 0.9855432211548814, 0.9855432211548814, 0.9855432211548814, 0.9962256383384895]
rnn_detect = [0.983962203213096, 0.983962203213096, 0.983962203213096, 0.983962203213096, 0.9944746119352343]
gru_detect = [0.9881608208989656, 0.9881608208989656, 0.9881608208989656, 0.9881608208989656, 0.9975700090908166]
lstm_detect = [0, 0, 0, 0, 0]
cnn_lstm_detect = [0.9892973697890233, 0.9892973697890233, 0.9892973697890233, 0.9892973697890233, 0.9980537147355995]

cnn_classify = [0.8949121855728845, 0.9005184649198067, 0.8949121855728845, 0.8921945143159461, 0.9941649909639073]
rnn_classify = [0.8748954590153388, 0.8799944594813578, 0.8748954590153388, 0.871594203366664, 0.992217493529757]
gru_classify = [0.9026459208337546, 0.9083398091756237, 0.9026459208337546, 0.8999618153355302, 0.9946449220919217]
lstm_classify = [0.9069194643825419, 0.9119750271341589, 0.9069194643825419, 0.9042075451351914, 0.9949896444633165]
cnn_lstm_classify = [0.8595428686965233, 0.8738038342974729, 0.8595428686965233, 0.8506560496526265, 0.9896879334417338]

app = Flask(__name__)

# API GET Models
@app.route('/api/models', methods=['GET'])
def get_data():
    model_types = {
        "classify": ["cnn", "rnn", "gru", "lstm", "cnn_lstm"],
        "detect": ["cnn", "rnn", "gru", "lstm", "cnn_lstm"]
    }

    response = []

    for model_type, models in model_types.items():
        models_info = []
        for model_name in models:
            model_score = globals()[f"{model_name}_{model_type}"]
            model_info = {
                'name': model_name,
                'info': {
                    'accuracy': model_score[0],
                    'precision': model_score[1],
                    'recall': model_score[2],
                    'f1_score': model_score[3],
                    'roc_auc': model_score[4]
            models_info.append(model_info)
        response.append({"models": models_info, "type": model_type})

    return jsonify(response), 200

# API POST Data (Check URL)
@app.route('/api/check', methods=['POST'])
def post_data():
    data = request.get_json()
    model_functions = {
        'classify': {
            'cnn': model_cnn_classify,
            'rnn': model_rnn_classify,
            'gru': model_gru_classify,
            'lstm': model_lstm_classify,
            'cnn_lstm': model_cnn_lstm_classify
        },
        'detect': {
            'cnn': model_cnn_detect,
            'rnn': model_rnn_detect,
            'gru': model_gru_detect,
            'lstm': model_lstm_detect,
            'cnn_lstm': model_cnn_lstm_detect
        }
    }
    model = model_functions.get(data.get('type'), {}).get(data.get('model'))
    if model:
        result = predict_url(data.get('domain'), model, data.get('type'))
        result = get_name_classify(result[0]) if data['type'] == 'classify' else get_name_detect(result[0])
        return jsonify({'result': result}), 200
    return jsonify({'error': 'Invalid request'}), 400

app.run(debug=True, port=5001)
