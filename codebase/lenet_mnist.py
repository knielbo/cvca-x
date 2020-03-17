#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of LeNet to mnist

Usage
$ python lenet_mnist.py
"""
from kartina.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
from kartina.kautil import gpu_fix

def main():
    gpu_fix()
    os.system("clear")

    # grab data
    print("[INFO] accessing MNIST...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # reshape if channels_first for theano users
        # design matrix is num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
        testData = testData.reshape((testData.shape[0], 1, 28, 28))
    
    # else num_samples x rows x columns x depth
    else:
        trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
        testData = testData.reshape((testData.shape[0], 28, 28, 1))
    
    # scale data [0, 1]
    trainData = trainData.astype("float32") / 255.
    testData = testData.astype("float32") / 255.

    # one-hot coding
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.fit_transform(testLabels)

    # init optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-2)
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # train model
    print("[INFO] training network...")
    H = model.fit(
        trainData,
        trainLabels,
        validation_data=(testData, testLabels),
        batch_size=128,
        epochs=20,
        verbose=1
    )

    # evaluate
    print("[INFO] evaluating network...")
    predictions = model.predict(testData, batch_size=128)
    print(
        classification_report(
            testLabels.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[str(x) for x in le.classes_]
        )
    )

    # plot performance
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(
        loc="center right", 
        ncol=2, fancybox=True,
        shadow=True
        )
    plt.tight_layout()
    plt.savefig("../figures/lenet_mnist.png")
    plt.close()

if __name__=="__main__":
    main()