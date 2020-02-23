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



if __name__=="__main__":
    main()
