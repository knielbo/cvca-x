#!/home/knielbo/virtenvs/cv/bin/python
"""
Train Perceptron class on OR
"""
from kartina.nn import Perceptron
import numpy as np

def main()
    # construct dataset
    X = np.array([0, 0], [0, 1], [1, 0], [0, 0])
    y = np.array([0], [1], [1], [1])

    # define and train perceptron
    print("[INFO] training perceptron...")
    p = Perceptron(X.shape[1], alpha=.1)
    p.fit(X, y, epochs=20)

    # evaluate
    print("[INFO] testing perceptron...")

    # loop over data points
    for (x, target) in zip(X, y):
        # predict and display
        pred = p.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))

