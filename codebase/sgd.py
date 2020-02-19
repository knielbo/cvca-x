#!/home/knielbo/virtenvs/cv/bin/python
"""
Implementing mini-batch Stochastic Gradient Descent
    - extension of gradient_descent.py with batches to reduce resource compsumption
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


def sigmoid_activation(x):
    """ sigmoid activation function/scoring function for input x """
    
    return 1.0 / (1 + np.exp(-x))

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

def next_batch(X, w, batchSize):
    # loop over dataset X in mini-batches and yield tuple of current batched data
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i+batchSize], y[i:i+batchSize])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
    ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
    ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batches")
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

    # loop for e epochs
    for epoch in np.arange(0, args["epochs"]):
        # dot product og X and W and feed through activation function to give model prediction
        
        preds = sigmoid_activation(trainX.dot(W))

        # determine error and loss using SSE
        error = preds - trainY
        loss = np.sum(error ** 2)
        losses.append(loss)

        # gradient descent update is the dot product between features X and the error of the sigmoid derivative
            # of the predictions
        d = error * sigmoid_deriv(preds)
        gradient = trainX.T.dot(d)
        
        # nudge the weight matrix in the negative direction of the gradient
        W += -args["alpha"] * gradient
        
        # check for diplay
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("[INFO] epoch={}, loss={:.7f}".format(int(epoch+1), loss))
        
    print("[INFO] evaluating...")
    preds = predict(testX, W)
    print(classification_report(testY, preds))

    # plot testing
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.title("$Data$")
    plt.scatter(testX[:,0],testX[:,1], marker="o",c=testY[:,0], s=30)
    plt.tight_layout()
    plt.savefig(os.path.join("..","figures","gradient_descent_data.png"))
    plt.close()

    # plot loss over time
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, args["epochs"]), losses)
    plt.title("$Training~Loss$")
    plt.xlabel("$Epoch #$")
    plt.ylabel("$Loss$")
    plt.tight_layout()
    plt.savefig(os.path.join("..","figures","gradient_descent_loss.png"))
    plt.close()






if __name__=="__main__":
    main()
