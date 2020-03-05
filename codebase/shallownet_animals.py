#!/home/knielbo/virtenvs/cv/bin/python
"""
ShallowNet application to 3-class animal dataset
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kartina.preprocessing import ImageToArrayPreprocessor
from kartina.preprocessing import SimplePreprocessor
from kartina.datasets import SimpleDatasetLoader
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

from kartina.kautil import gpu_fix
import os

def main():
    """
    ###################################################################################
    # bug fix for RTX architecture
        # https://github.com/tensorflow/tensorflow/issues/24496
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    ###################################################################################
    """
    gpu_fix()
    os.system("clear")
    # argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())

    # grap list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
    
    # initialize image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load datafrom and scale pixel intensities to [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.

    # partition data set 75/25
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.25, random_state=42)

    # encode lavels as vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # initialize optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), 
        batch_size=32, epochs=100, verbose=1)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(
        classification_report(
            testY.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=["cat", "dog", "panda"]
            )
        )

    # plot performance
    plt.style.use("fivethirtyeight")    
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figures/shallownet_animals.png")
    plt.close()


if __name__=="__main__":
    main()
