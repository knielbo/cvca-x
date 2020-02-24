#!/home/knielbo/virtenvs/cv/bin/python
"""
Classification of CIFAR-10 with Fully-Connected feedforward neural network
"""
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    # load data, scale and reshape
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float")/255.0
    testX = testX.astype("float")/255.0
    trainX = trainX.reshape((trainX.shape[0], 3072))
    testX = testX.reshape((testX.shape[0], 3072))

    # encode labels as one-hot vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # initialize label names for CIFAR-10
    labelNames = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
         ]

    # define 3072-1024-512-10
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # train model with SGD
    print("[INFO] training network...")
    sgd = SGD(1e-2)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)
    
    

if __name__=="__main__":
    main()