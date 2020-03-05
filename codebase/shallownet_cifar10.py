#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet application to cifar10 dataset
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from kartina.kautil import gpu_fix
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    gpu_fix()
    os.system("clear")

    # load data and scale [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.
    testX = testX.astype("float") / 255.

    # integers to one-hot vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # initialize label names for CIFAR-10 dataset
    labelNames = ["airplane", "automobile", "bird", "cat", "dear",
        "dog", "frog", "horse", "ship", "truck"]


if __name__=="__main__":
    main()