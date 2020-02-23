#!/home/knielbo/virtenvs/cv/bin/python
"""
Train feedforward NN on mnist (full) as baseline with Keras & TF
"""
import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def main():
    # user input
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True, help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    # grab mnist dataset (11MB)
    print("[INFO] accessing MNIST...")
    ((trainX, trainY), (testX, testY)) = mnist.load_data()
    
    # each image is 28x28x1 (grayscale)
    # flatten
    trainX = trainX.reshape((trainX.shape[0], 28*28 * 1))
    testX = testX.reshape((testX.shape[0], 28*28 * 1))

    # rescale grayscale light intensities to [0, 1]
    trainX = trainX.astype("float32") / 255.
    testX = testX.astype("float32") / 255.

    # convert labels to one-hot encoding
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # define architecture 784x256x128x10
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))

    # train model using SGD
    print("[INFO] training network...")
    sgd = SGD(1e-2)#learning rate
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX,testY), epochs=100, batch_size=128)
    
    # evaluate network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
        target_names=[str(x) for x in lb.classes_]))
    
    # plot training loss and accuracy
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(args["output"])


if __name__ == "__main__":
    main()


