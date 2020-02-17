#!/home/knielbo/virtenvs/cv/bin/python
"""
Implementing basic gradient descent
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    """ sigmoid activation function/scoring function for input x """
    
    return 1.0 / (1 + np.exp(x))

def sigmoid_deriv(x):
    """ compute derivate of sigmoid assuming that x has be passed through the sigmoid function """
    
    return x * (1 - x)

def predict(X, W):
    """ thresholded binary prediction """
    # dot product
    preds = sigmoid_activation(X.dot(W))
    # threshold
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
    ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
    args = vars(ap.parse_args())

    # generate classification problem n=1000, 2d feature matrix
    (X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
    y = y.reshape((y.shape[0], 1))

    # insert a column of 1's as last entry in the feature matrix -> treat bias as embedded trainable parameter
        # within the weight matrix
    X = np.c_[X, np.ones(X.shape[0])]

    # partition data
    (trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)
    # initialize W and list losses
    print("[INFO] training...")
    W = np.random.randn(X.shape[1], 1)
    losses = list()


if __name__=="__main__":
    main()
