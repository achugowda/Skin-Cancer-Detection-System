"""
    This file used to evaluate our model for fine-tuning and better understanding.
    It shows the cunfusion matrix, accuracy & loss histogram and classification report.
"""

import numpy as np
import tensorflow.keras as K
import pickle
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

x_test = pickle.load(open('/content/drive/MyDrive/skin_cancer_data/x_test', 'rb'))
y_test = pickle.load(open('/content/drive/MyDrive/skin_cancer_data/y_test', 'rb'))

hist_path = "/content/drive/MyDrive/skin_cancer_data/second/hist"
with open(hist_path, "rb") as file_pi:
    hist = pickle.load(file_pi)

model_path = "/content/drive/MyDrive/skin_cancer_data/second_train/max_acc.keras"  # or "/content/drive/MyDrive/skin_cancer_data/second_train/max_acc.keras"
model = K.models.load_model(model_path)

predictions = model.predict(x_test)
test_pred = np.argmax(predictions, axis=1)

types = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

cm = confusion_matrix(y_test, test_pred)                    # confusion matrix
print("Confusion Matrix\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=types)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.show()

acc_arr, val_acc_arr = [0.0], [0.0]
for i in hist['accuracy']:
    acc_arr.append(i)
for i in hist['val_accuracy']:
    val_acc_arr.append(i)
plt.plot(acc_arr)                               # plot accuracy vs epoch
plt.plot(val_acc_arr)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xticks(np.arange(0, 25, 2))
plt.grid()
plt.show()

loss_arr, val_loss_arr = [0.0], [0.0]
for i in hist['loss']:
    loss_arr.append(i)
for i in hist['val_loss']:
    val_loss_arr.append(i)
plt.plot(loss_arr)                              # Plot loss values vs epoch
plt.plot(val_loss_arr)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xticks(np.arange(0, 25, 2))
plt.grid()
plt.show()

for i in range(6):
    print(f'{types[i]} - {((cm[i][i] / sum(cm[i])) * 100):.2f}%')

print("\nclassification_report: \n" + str(classification_report(y_test, test_pred)))