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

model_path = 'model.h5'
image_path = '8865_idx5_x1651_y651_class1.png'

dict_characters = {
    0: 'IDC-',
    1: 'IDC+',
}

predict_and_plot(model_path, image_path, dict_characters)
