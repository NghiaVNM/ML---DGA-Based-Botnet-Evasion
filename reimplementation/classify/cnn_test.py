from keras.models import load_model
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# # test 1
# test1 = pd.read_csv('../dataset/classify/test1.txt', header=None)

# test1labels = pd.read_csv('../dataset/classify/test1label.txt', header=None)

# T1 = test1.values.tolist()
# T1 = list(itertools.chain(*T1))

# # test 2
# test2 = pd.read_csv('../dataset/classify/test2.txt', header=None)

# test1label = pd.read_csv('../dataset/classify/test2label.txt', header=None)

# T2 = test2.values.tolist()
# T2 = list(itertools.chain(*T2))

# Generate a dictionary of valid characters
valid_chars = {'w': 1, 'f': 2, '6': 3, 's': 4, 'v': 5, '3': 6, 'x': 7, 'p': 8, 'i': 9, 'm': 10, 'd': 11, '2': 12, 'c': 13, '1': 14, '4': 15, 'a': 16, 'q': 17, '0': 18, '.': 19, 'u': 20, 'b': 21, '_': 22, '-': 23, 'n': 24, 'j': 25, '7': 26, '8': 27, '5': 28, 't': 29, 'o': 30, 'k': 31, 'g': 32, '9': 33, 'l': 34, 'y': 35, 'r': 36, 'e': 37, 'z': 38, 'h': 39}

max_features = 40

maxlen = 91

# # Convert characters to int and pad
# T1 = [[valid_chars[y] for y in x] for x in T1]
# X_test1 = sequence.pad_sequences(T1, maxlen=maxlen)

# Load the best model
best_model = load_model('./logs/cnn/checkpoint-00.h5')
# Prepare data for new domain
new_domain = "fiddlrxsat.com"
encoded_new_domain = [[valid_chars[y] for y in new_domain]]
padded_new_domain = sequence.pad_sequences(encoded_new_domain, maxlen=maxlen)
print(best_model.predict(padded_new_domain))
# Predict classification of new domain
predicted_class = np.argmax(best_model.predict(padded_new_domain), axis=-1)
print("Predicted class for", new_domain, ":", predicted_class)
# predicted_classes = np.argmax(best_model.predict(X_test1), axis=-1)

# # Calculate metrics
# accuracy = accuracy_score(test1labels, predicted_classes)
# precision = precision_score(test1labels, predicted_classes, average='weighted')
# recall = recall_score(test1labels, predicted_classes, average='weighted')
# f1 = f1_score(test1labels, predicted_classes, average='weighted')
# conf_matrix = confusion_matrix(test1labels, predicted_classes)

# # Save results
# with open('./logs/cnn/evaluation_results_test1.txt', 'w') as f:
#     f.write("Accuracy: {}\n".format(accuracy))
#     f.write("Precision: {}\n".format(precision))
#     f.write("Recall: {}\n".format(recall))
#     f.write("F1 Score: {}\n".format(f1))
#     f.write("Confusion Matrix:\n{}\n".format(conf_matrix))

# fpr, tpr, thresholds = roc_curve(test1labels, predicted_probabilities[:, 1])
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig('./logs/cnn/roc_curve_test1.png')
# plt.show()