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
valid_chars = {
'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 
'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '0': 27, '1': 28, '2': 29, '3': 30,
'4': 31, '5': 32, '6': 33, '7': 34, '8': 35, '9': 36, '.': 37, '-': 38, '_': 39, 'ɑ': 40,
'ằ': 41, 'ӑ': 42, 'ä': 43, 'ǟ': 44, 'ā': 45, 'ậ': 46, 'ą': 47, 'ẩ': 48, 'ả': 49, 'à': 50,
'ă': 51, 'ǎ': 52, 'ặ': 53, 'á': 54, 'ã': 55, 'ạ': 56, 'å': 57, 'ầ': 58, 'ȃ': 59, 'а': 60,
'ắ': 61, 'a': 62, 'ḁ': 63, 'ȧ': 64, 'ά': 65, 'ἀ': 66, 'α': 67, 'ƅ': 68, 'b': 69, 'ʙ': 70,
'ḃ': 71, 'ɓ': 72, 'Ꮟ': 73, 'ᖯ': 74, 'ᑲ': 75, 'ḅ': 76, 'ь': 77, 'ƈ': 78, 'ⲥ': 79, 'ć': 80,
'c': 81, 'č': 82, 'ᴄ': 83, '𐐽': 84, 'с': 85, 'd': 86, 'ḍ': 87, 'ḓ': 88, 'đ': 89, 'ꓒ': 90,
'ḑ': 91, 'ᑯ': 92, 'ԁ': 93, 'ɗ': 94, 'Ꮷ': 95, 'ḋ': 96, 'ẽ': 97, 'ễ': 98, 'ё': 99, 'є': 100,
'ɛ': 101, 'è': 102, 'ê': 103, 'ế': 104, 'ể': 105, 'ệ': 106, 'е': 107, 'ɘ': 108, 'ë': 109, 'é': 110,
'ҽ': 111, 'ḛ': 112, 'ẹ': 113, 'ĕ': 114, 'ȩ': 115, 'e': 116, 'ę': 117, 'ě': 118, 'ε': 119, 'ė': 120,
'ē': 121, 'ḟ': 122, 'ƒ': 123, 'ꬵ': 124, 'ẝ': 125, 'ꞙ': 126, 'ғ': 127, 'f': 128, 'ǵ': 129, 'ց': 130,
'g': 131, 'ğ': 132, 'ġ': 133, 'ĝ': 134, 'ɡ': 135, 'ģ': 136, 'Ꮒ': 137, 'һ': 138, 'ḥ': 139, 'h': 140,
'հ': 141, 'ĥ': 142, 'i': 143, 'ï': 144, 'î': 145, 'ӏ': 146, 'Ꭵ': 147, 'ꙇ': 148, 'ì': 149, 'і': 150,
'ị': 151, 'ǐ': 152, 'ї': 153, 'ı': 154, 'ī': 155, 'ɩ': 156, 'ỉ': 157, 'í': 158, 'ί': 159, 'ɪ': 160,
'ι': 161, 'ȋ': 162, 'į': 163, 'j': 164, 'ϳ': 165, 'ĵ': 166, 'ј': 167, 'ḵ': 168, 'к': 169, 'k': 170,
'κ': 171, 'ⱪ': 172, 'ĸ': 173, 'ķ': 174, 'ᴋ': 175, 'ƙ': 176, 'ḳ': 177, 'ł': 178, 'ᛁ': 179, 'ꞁ': 180,
'ⵏ': 181, 'ǀ': 182, 'ľ': 183, '𐌉': 184, 'ꓲ': 185, 'ĩ': 186, 'l': 187, 'ṃ': 188, 'm': 189, 'м': 190,
'ḿ': 191, 'ᴍ': 192, 'ṁ': 193, 'ñ': 194, 'ṅ': 195, 'ṉ': 196, 'п': 197, 'n': 198, 'ń': 199, 'ň': 200,
'ŋ': 201, 'ņ': 202, 'ǹ': 203, 'ꞑ': 204, 'ɳ': 205, 'η': 206, 'ո': 207, 'ή': 208, 'ƞ': 209, 'ṇ': 210,
'ō': 211, 'о': 212, '൦': 213, '၀': 214, '೦': 215, 'ό': 216, 'ơ': 217, 'ᴏ': 218, '௦': 219, '໐': 220,
'ỡ': 221, 'ờ': 222, 'ഠ': 223, '๐':  224,  ' ȯ': 225, 'օ': 226, 'õ': 227, 'o': 228, '०': 229, 'ဝ': 230,
'੦': 231, 'σ': 232, 'ö': 233, 'ò': 234, 'ჿ': 235, '౦': 236, 'ổ': 237, 'ồ': 238, '૦': 239, 'ő': 240,
'ó': 241, 'ợ': 242, 'ŏ': 243, 'ᴑ': 244, 'ο': 245, 'ọ': 246, 'ǫ': 247, 'ⲟ': 248, 'ộ': 249, 'ӧ': 250,
'ṗ': 251, 'ⲣ': 252, 'p': 253, 'р': 254, 'ρ': 255, 'q': 256, 'զ': 257, 'ԛ': 258, 'գ': 259, 'г': 260,
'ř': 261, 'ⲅ': 262, 'ꭇ': 263,  'ᴦ': 264, 'ʀ': 265, 'ɾ': 266, 'ꭈ': 267, 'ṛ': 268, 'ŗ': 269, 'ṙ': 270,
'r': 271, 's': 272, 'ṡ': 273, 'ʂ': 274, 'ṣ': 275, 'ѕ': 276, '𐑈': 277, 'š': 278, 'ś': 279, 'ș': 280,
'ꜱ': 281, 'ť': 282, 'ṫ': 283, 'ʈ': 284, 'ț': 285, 't': 286, 'ṭ': 287, 'т': 288, 'ŧ': 289, 'ƫ': 290,
'ữ': 291, 'ŭ': 292, 'ứ': 293, 'ư': 294, 'ǔ': 295, 'ʋ': 296, 'ǚ': 297, '𐓶': 298, 'ụ': 299, 'ử': 300,
'û': 301, 'ũ': 302, 'ǜ': 303, 'υ': 304, 'u': 305, 'ű': 306, 'ᴜ': 307, 'ꭒ': 308, 'ü': 309, 'ù': 310,
'ս': 311, 'ū': 312, 'ύ': 313, 'ꭎ':  314,  'v': 315,  'ν': 316, 'ᴠ': 317, 'ṿ': 318, 'ѵ': 319, 'w': 320,
'ẁ': 321, 'ա': 322, 'ᴡ': 323, 'ω': 324, 'ш': 325, 'ẘ': 326, 'ẉ': 327, 'ԝ': 328, 'ŵ': 329, 'ẇ': 330,
'ɯ': 331, 'ẃ': 332, 'ⱳ': 333, 'ѡ': 334, 'χ': 335, 'ẍ': 336, 'х': 337, 'ẋ': 338, 'ᕁ': 339, 'x': 340,
'ᕽ': 341, 'ɣ': 342, 'ŷ': 343, 'y': 344, 'у': 345, 'ỳ': 346, 'ү': 347, 'ყ': 348, 'ỵ': 349, 'γ': 350,
'ʏ': 351, 'ý': 352, 'ỷ': 353, 'ÿ': 354, 'ƴ': 355, 'ỹ': 356, 'ž': 357, 'ź': 358, 'ẕ': 359, 'ᴢ': 360,
'z': 361, 'ż': 362, ',': 363, 'A': 364, 'B': 365, 'C': 366, 'D': 367, 'E': 368, 'F': 369, 'G': 370,
'H': 371, 'I': 372, 'J': 373, 'K': 374, 'L': 375, 'M': 376, 'N': 377, 'O': 378, 'P': 379, 'Q': 380,
'R': 381, 'S': 382, 'T': 383, 'U': 384, 'V': 385, 'W': 386, 'X': 387, 'Y': 388, 'Z': 389
}

max_features = 390

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
                    epochs=5, batch_size=32, 
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
