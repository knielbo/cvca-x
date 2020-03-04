#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet application to 3-class animal dataset
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kartina.preprocessing import ImageToArrayPreprocessor
from kartina.preprocessing import SimplePreprocessor
from kartina.datasets import SimpleDatasetLoader
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    # argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())

    # grap list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    
    # initialize image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load datafrom and scale pixel intensities to [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, label) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.

if __name__=="__main__":
    main()
