#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet application to cifar10 dataset
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from kartina.kautil import gpu_fix
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    gpu_fix()
    os.system("clear")

    # load data and scale [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.
    testX = testX.astype("float") / 255.

    # integers to one-hot vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # initialize label names for CIFAR-10 dataset
    labelNames = ["airplane", "automobile", "bird", "cat", "dear",
        "dog", "frog", "horse", "ship", "truck"]

    # initialize optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr =.01)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    # train network
    print("[INFO] training network...")
    H = model.fit(
        trainX, trainY, 
        validation_data=(testX, testY), 
        batch_size=32,
        epochs=40,
        verbose=1
        )
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(
        classification_report(
            testY.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=labelNames
            )
        )

    # visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figures/shallownet_cifar10.png")
    plt.close()

if __name__=="__main__":
    main()