#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of ShallowNet to animals3
    - illustrate load methods for trained models
Usage:
$ python shallownet_load.py --dataset ../datasets/animals --model ../models/shallownet_weights.hdf5

"""
from kartina.preprocessing import ImageToArrayPreprocessor
from kartina.preprocessing import SimplePreprocessor
from kartina.datasets import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from kartina.kautil import gpu_fix



def main():
    # bug fixing
    gpu_fix()
    os.system("clear")

    # argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    args = vars(ap.parse_args())

    # class labels
    classLabels = ["cat", "dog", "panda"]

    # sample from list of images in dataset
    print("[INFO] sampling images...")
    imagePaths = np.array(
        list(
            paths.list_images(args["dataset"])
        )
    )
    idxs = np.random.randint(0, len(imagePaths), size=(5,))
    imagePaths = imagePaths[idxs]

    # initialize preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load and scale data set to range [0,1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths)
    data = data.astype("float") / 255.

    # load pre-trained model
    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    # predict image class
    print("[INFO] predicting")

    

if __name__=="__main__":
    main()
