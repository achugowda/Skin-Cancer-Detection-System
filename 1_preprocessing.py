"""
    This file used to load the entire dataset, perform the require preprocessing and split the data into train, validation and test sets
"""

import pandas as pd
import os
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import img_to_array

img_path = os.listdir('/content/drive/MyDrive/skin_cancer_data/HAM10000_images_part_1')

features = []
labels = []
image_labels = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
for i in tqdm(img_path):
    fname = os.path.join('/content/drive/MyDrive/skin_cancer_data/HAM10000_images_part_1', i)
    img = image.load_img(fname, target_size=(224, 224))
    x = img_to_array(img)
    x = K.applications.xception.preprocess_input(x)
    features.append(x)
    image_id = i.split(".")[0]
    label = metadata.loc[metadata['image_id'] == image_id, 'dx'].iloc[0]
    labels.append(image_labels[label])

features = np.array(features)
labels = np.array(labels)

lbl, count = np.unique(labels, return_counts=True)                                            # plot a bar plot to show the quantity of images in each class
temp = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
count_lbl = {}
for i in range(len(lbl)):
    count_lbl[temp[lbl[i]]] = count[i]
sns.set_theme(style="whitegrid")
ax = sns.barplot(x=list(count_lbl.keys()), y=list(count_lbl.values()))
for i in ax.containers:
    ax.bar_label(i,)
plt.title('Before augmentation')
plt.show()

indices = np.random.permutation(len(features))
features = features[indices]
labels = labels[indices]

x_train, y_train, rx_test, ry_test, x_val, y_val, x_test, y_test = [], [], [], [], [], [], [], []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)       # split the test set
for train_index, r_index in split.split(features, labels):
    x_train, rx_test = features[train_index], features[r_index]
    y_train, ry_test = labels[train_index], labels[r_index]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)      # split the training data into training and validation sets
for val_index, test_index in split.split(rx_test, ry_test):
    x_val, x_test = rx_test[val_index], rx_test[test_index]
    y_val, y_test = ry_test[val_index], ry_test[test_index]

f = open("/content/drive/MyDrive/skin_cancer_data/x_train", "wb")
pickle.dump(x_train, f)
f.close()
f = open("/content/drive/MyDrive/skin_cancer_data/y_train", "wb")
pickle.dump(y_train, f)
f.close()

f = open("/content/drive/MyDrive/skin_cancer_data/x_val", "wb")
pickle.dump(x_val, f)
f.close()
f = open("/content/drive/MyDrive/skin_cancer_data/y_val", "wb")
pickle.dump(y_val, f)
f.close()

f = open("/content/drive/MyDrive/skin_cancer_data/x_test", "wb")
pickle.dump(x_test, f)
f.close()
f = open("/content/drive/MyDrive/skin_cancer_data/y_test", "wb")
pickle.dump(y_test, f)
f.close()