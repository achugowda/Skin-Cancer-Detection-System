"""
    This file used to load our model and predict uploaded image.
"""

from IPython.display import display, Javascript, HTML
from google.colab import files
from base64 import b64decode
import io
import os
import numpy as np
from tqdm import tqdm
import tensorflow.keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from google.colab import output

# Function to create an upload button and handle image uploads
def upload_image():
  uploaded = files.upload()

  for filename in uploaded.keys():
    img = image.load_img(io.BytesIO(uploaded[filename]), target_size=(224, 224))
    img_array = img_to_array(img)
    x = K.applications.xception.preprocess_input(img_array)

    prediction = model.predict(np.array([x]))[0]
    test_pred = np.argmax(prediction)

    result = [(types[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
    result.sort(reverse=True, key=lambda x: x[1])

    output_html = "<h2>Prediction Results:</h2>"
    for j in range(6):
      (class_name, prob) = result[j]
      output_html += f"<p>Top {j + 1} ====================</p>"
      output_html += f"<p>{class_name}: {prob:.2f}%</p>"

    display(HTML(output_html))

# Load the model and define the types
model = K.models.load_model("/content/drive/MyDrive/skin_cancer_data/second_train/max_acc.keras")
types = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

upload_image()
output.register_callback('notebook.upload_image', upload_image)