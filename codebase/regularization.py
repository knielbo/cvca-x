#!/home/knielbo/virtenvs/cv/bin/python
"""
Test L1 and L2 regularization
"""
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from kartina.preprocessing import SimplePreprocessor
from kartina.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def main():
    # construct argument parse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())

    # grab list of image paths
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize image preprocessor, load and reshape
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))
    
    # encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # split data set
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.25, random_state=42)

    
    # loop over regularizers
    for r in (None, "l1", "l2"):
        # train SGD classifier with softmax loss and regularization for 10 epocs
        print("[INFO] training model with `{}` penalty".format(r))
        model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant", tol=1e-3, eta0=.01, random_state=42)
        model.fit(trainX, trainY)

        # evaluate the classifier
        acc = model.score(testX, testY)
        print("[INFO] `{}` penalty accuracy: {:.2f}%".format(r, acc * 100))


if __name__=="__main__":
    main()

