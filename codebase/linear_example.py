#!/home/knielbo/virtenvs/cv/bin/python
"""
Example of linear classifier for parameterized learning
    - no learning, hardcoded solution to explain Scoring Function
"""
import os
import numpy as np
import cv2


def main():
    # initialize class labels and seed
    labels = ["dog", "cat", "panda"]
    np.random.seed(1)
    # randomly initialize weights and bias
    W = np.random.randn(3, 3072)
    b = np.random.randn(3)
    # load example
    orig = cv2.imread(os.path.join("..", "datasets", "beagle.png"))
    image = cv2.resize(orig, (32,32)).flatten()
    # SCORING FUNCTINO: compute output score as inner prodoct of weight matrix and image pixels adding the bias
    scores = W.dot(image) + b
    # loop over scores and labels
    for (label, score) in zip(labels, scores):
        print("[INFO] {}: {:.2f}".format(label, score))
    # draw label of argmax on image
    cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    # display image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)

if __name__=="__main__":
    main()