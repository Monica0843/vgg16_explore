# USAGE
# python predict.py

# imports package necessary for feature extraction 
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils
from keras.layers import Input

# other imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from pyimagesearch import config
import sys 
import numpy as np
import os
import json
import pickle
import cv2

def predict():
    script = sys.argv[0]
    filename = sys.argv[1]
    modelname = sys.argv[2]

    # extract features from the target image
    model = VGG16(weights="imagenet", include_top=False)
    image = load_img(filename, target_size=(224,224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    features = model.predict(x)
    features = features.reshape((features.shape[0], 7*7*512))

    # load the train model from disk
    model = pickle.loads(open(modelname,"rb").read())
    y = model.predict(features)

    print("Prediction Model used is:", "%s" %modelname)


    # use LabelEncoder inverse_transform to interpret the result
    # load the labelencoder from disk
    le = pickle.loads(open(config.LE_PATH, "rb").read())

    # first convert the model output y from an array to a list format
    # as the input of labelencoder
    y = [int(y[0])]
    result = le.inverse_transform(y)

    # print the final result
    if result[0] == "food":
        print("It is food.")
    else:
        print("It is not food.")
        
predict()
