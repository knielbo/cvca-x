#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of ShallowNet to animals3
    - illustrate save methods for trained models
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kartina.preprocessing import SimplePreprocessor
from kartina.preprocessing import ImageToArrayPreprocessor
from kartina.datasets import SimpleDatasetLoader
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from kartina.kautil import gpu_fix

def main():
    gpu_fix()
    os.system("clear")

    # argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    args = vars(ap.parse_args())

    # grab list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load data and scale to [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=5e2)
    data = data.astype("float") / 255.

    # partition
    (trainX, testX, trainY, testY) = train_test_split(
            data, labels, 
            test_size=.25, 
            random_state=42
            )
    



if __name__=="__main__":
    main()