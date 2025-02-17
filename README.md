# Skin-Cancer-Detection-System
Welcome to Skin Cancer Detection System, an advanced system that leverages machine learning and image processing to identify and diagnose six different types of skin lesions, providing valuable assistance with the help of chatbot.

Introduction: Skin conditions are a frequent cause of clinic visits, where accurate diagnosis is essential for effective treatment. This project introduces a powerful machine learning system that analyzes images to detect and classify various skin lesions.

Dataset: The project uses the HAM10000(Human Against Machine with 10000 trainingimages)dataset for skin cancer detection. This dataset containsdermatoscopic images of pigmented skin lesions across7different diagnostic categories.

Model: The skin cancer detection project utilizes the Xception architecture as its base model. Xception, short for "Extreme Inception," is a deep convolutional neural network

Platforms used: google colab, Spyder, Streamlit

Repository Structure

'preprocessing.py': This code loads the entire dataset, perform the required image preprocessing, and splits the images into train, validation and test sets.

'visualization.py': This code used to show the distribution of the different skin lesions' types through the train, validation and test sets.

'augmentation.py': Code for adding augmented images to our dataset for classes with a lack of images.

'modelBuild.py': The code we used to build our Xception model for skin lesion diagnosis.

'evaluate.py': Code for evaluating our model for fine-tuning and better understanding. It shows the confusion matrix, accuracy and loss histograms, and classification report.

'predict.py': Code for prediction a batch of images from a directory, using our model.

'saveModel.py': This code saves a trained model to a file using pickle and later loads it for reuse.

'webApp.py': Running this code in Spyder will launch a web page on the Streamlit platform.
