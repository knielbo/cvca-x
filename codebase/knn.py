#!/home/knielbo/virtenvs/cv/bin/python
"""
Train end-to-end image classifier using k-Nearest Neighbors

Input:
    [required]
        -d: path to dataset (folder) using /path/to/dataset/{class}/{image}.jpg convention
    [optional]
        -k: number of nearest neighbors (default 1)
        -j: number of jobs (default -1, all cores)

Example:
    python knn.py --dataset ../datasets/animals --neighbors 2 --jobs 4
"""
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from kartina.preprocessing import SimplePreprocessor
from kartina.datasets import SimpleDatasetLoader


def main():
    # construct the argument parse and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
    args = vars(ap.parse_args())
    # grab the list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the image preprocessor, load dataset, and reshape data matrix
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))
    # show some information on memory consumption of the images
    print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024*1024.0)))
    # encode labes as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # partition data set
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.25, random_state=23)
    # train and evaluate k-NN classifier on raw pixel intensities (end-to-end model)
    print("[INFO] evaluating k-NN classifier...")
    model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
    model.fit(trainX, trainY)
    print(classification_report(testY, model.predict(testX), target_names=le.classes_))

if __name__=="__main__":
    main()