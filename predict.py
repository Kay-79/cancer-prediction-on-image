import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D

def predict_and_plot(model_path, image_path, class_dict):
    """
    Load a model, predict the class of an image, and plot the result.
    """
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (50, 50))  # Resize to match model input
    image_normalized = image_resized / 255.0  # Normalize pixel values
    image_reshaped = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(image_reshaped)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]

    # Plot the image and prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Prediction: {class_dict[predicted_class]} ({confidence:.2f})")
    plt.show()

def evaluate_and_plot(model_path, folder_path, class_dict):
    """
    Evaluate predictions for all images in a folder, display predictions, and show a bar chart.
    """
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Initialize counters for predictions
    predictions_count = {label: 0 for label in class_dict.values()}

    # Prepare to display images and predictions
    image_paths = glob(os.path.join(folder_path, "*.png"))
    num_images = len(image_paths)
    cols = 5
    rows = (num_images // cols) + (num_images % cols > 0)

    plt.figure(figsize=(cols * 3, rows * 3))  # Dynamically adjust figure size
    for idx, image_path in enumerate(image_paths):
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (50, 50))
        image_normalized = image_resized / 255.0
        image_reshaped = np.expand_dims(image_normalized, axis=0)

        # Predict the class
        prediction = model.predict(image_reshaped)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        predictions_count[class_dict[predicted_class]] += 1

        # Extract the last 6 characters of the filename
        filename_suffix = os.path.basename(image_path)[-10:]

        # Display the image and prediction
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"{class_dict[predicted_class]} ({confidence:.2f})\n{filename_suffix}")

    plt.tight_layout()
    plt.show()

    # Plot the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(predictions_count.keys(), predictions_count.values(), color=['blue', 'orange'])
    plt.title("Prediction Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# Example usage:
model_path = 'model.h5'
# image_path = '16534_idx5_x151_y101_class1.png'
test_folder_path = 'test'  # Folder containing test images

dict_characters = {
    0: 'IDC-',
    1: 'IDC+',
}

# predict_and_plot(model_path, image_path, dict_characters)
evaluate_and_plot(model_path, test_folder_path, dict_characters)
