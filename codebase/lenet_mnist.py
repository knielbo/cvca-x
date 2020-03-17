#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of LeNet to mnist

Usage
$ python
"""
from kartina.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from kartina.kautil import gpu_fix

def main():
    gpu_fix()
    os.system("clear")

    # grab data
    print("[INFO] accessing MNIST...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # reshape if channels_first for theano users
        # design matrix is num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
        testData = testData.reshape((testData.shape[0], 1, 28, 28))
    
    # else num_samples x rows x columns x depth
    else:
        trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
        testData = testData.reshape((testData.shape[0], 28, 28, 1))

if __name__=="__main__":
    main()