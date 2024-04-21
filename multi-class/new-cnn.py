from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from sklearn.preprocessing import Normalizer
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Convolution1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import itertools

np.set_printoptions(threshold=np.inf)

# Train data
trains = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
train = trains.iloc[:,0:1]
trainlabels = pd.read_csv('../dataset/dgcorrect-multi/trainlabel-multi.csv', header=None)
trainlabel = trainlabels.iloc[:,1:2]

X = train.values.tolist()
X = list(itertools.chain(*X))

# Generate a dictionary of valid characters
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
max_features = len(valid_chars) + 1
maxlen = np.max([len(x) for x in X])

# Convert characters to int and pad
X1 = [[valid_chars[y] for y in x] for x in X]
X_train = sequence.pad_sequences(X1, maxlen=maxlen)

y_trainn = np.array(trainlabel)
y_train = to_categorical(y_trainn)

from keras.layers import Conv1D, MaxPooling1D, Flatten

hidden_dims=128
nb_filter = 32
filter_length =3 
embedding_vecor_length = 128
num_classes = 21

# Khởi tạo mô hình
model = Sequential()

# Thêm lớp Embedding
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))

# Thêm lớp Convolutional
model.add(Conv1D(nb_filter, filter_length, padding='valid', activation='relu'))

# Thêm lớp MaxPooling
model.add(MaxPooling1D())

# Thêm lớp Flatten để chuyển từ tensor 3D sang vector 1D
model.add(Flatten())

# Thêm lớp Dense (fully connected)
model.add(Dense(hidden_dims, activation='relu'))

# Lớp đầu ra
model.add(Dense(num_classes, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Hiển thị cấu trúc mô hình
print(model.summary())

from sklearn.model_selection import train_test_split

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# In ra biểu đồ về sự thay đổi của độ chính xác và hàm mất mát qua các epoch
import matplotlib.pyplot as plt

# Lấy thông tin về độ chính xác và hàm mất mát từ quá trình huấn luyện
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Vẽ biểu đồ
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Chuẩn bị dữ liệu cho domain mới
new_domain = "elelg850k4qdgf5x7lgvoxy.ddns.net"  # Thay đổi domain mới tại đây

# Mã hóa domain mới thành vectơ số nguyên
encoded_new_domain = [[valid_chars[y] for y in new_domain]]

# Sử dụng hàm pad_sequences để đảm bảo độ dài phù hợp
padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)

# Dự đoán phân loại của domain mới
predicted_class = np.argmax(model.predict(padded_new_domain), axis=-1)

# In kết quả dự đoán
print("Predicted class for", new_domain, ":", predicted_class)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# Dự đoán phân loại trên tập kiểm tra
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(np.argmax(y_test, axis=-1), y_pred_classes)
precision = precision_score(np.argmax(y_test, axis=-1), y_pred_classes, average='weighted')
recall = recall_score(np.argmax(y_test, axis=-1), y_pred_classes, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=-1), y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(np.argmax(y_test, axis=-1), y_pred_classes)

# Vẽ ROC curve và tính toán AUC
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

# In kết quả
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
print("ROC AUC:", roc_auc)

# Vẽ ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
