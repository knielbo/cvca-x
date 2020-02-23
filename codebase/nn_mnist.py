#!/home/knielbo/virtenvs/cv/bin/python
"""
Train NeuralNetwork instance on mnist (sample) as baseline
"""
from kartina.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

def main():
    # load data and apply min/max scaling
    # each image is 8x8 pixels grayscale
    print("[INFO] loading MNIST (sample) dataset")
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min())/(data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

    # split data
    (trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=.25)

    # convert labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # train network
    print("[INFO] training network...")
    nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
    print("[INFO] {}".format(nn))
    nn.fit(trainX, trainY, epochs=1000)

    # evaluate network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(testX)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions))


if __name__=="__main__":
    main()
