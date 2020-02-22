#!/home/knielbo/virtenvs/cv/bin/python
"""
Train NeuralNetwork class on XOR
"""
from kartina.nn import NeuralNetwork
import numpy as np

def main():
    # construct dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # define and train perceptron
    print("[INFO] training neural network...")
    nn = NeuralNetwork([2, 2, 1], alpha=.5)
    ## test without hidden layer/update
    #nn = NeuralNetwork([2, 1], alpha=.5)
    nn.fit(X, y, epochs=20000)

    # evaluate
    print("[INFO] testing neural network...")

    # loop over data points
    for (x, target) in zip(X, y):
        # predict and display
        pred = nn.predict(x)[0][0]
        step = 1 if pred > .5 else 0
        print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))

if __name__=="__main__":
    main()