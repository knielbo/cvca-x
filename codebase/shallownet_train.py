#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of ShallowNet to animals3
    - illustrate save methods for trained models
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kartina.preprocessing import SimplePreprocessor
from kartina.preprocessing import ImageToArrayPreprocessor
from kartina.datasets import SimpleDatasetLoader
from kartina.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from kartina.kautil import gpu_fix

def main():
    gpu_fix()
    os.system("clear")

    # argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    args = vars(ap.parse_args())

    # grab list of images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()

    # load data and scale to [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(imagePaths, verbose=5e2)
    data = data.astype("float") / 255.

    # partition
    (trainX, testX, trainY, testY) = train_test_split(
            data, labels, 
            test_size=.25, 
            random_state=42
            )
    
    # one-hot encoding
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    
    # init optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=5e-3)
    model = ShallowNet.build(
        width=32,
        height=32,
        depth=3,
        classes=3
        )
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
        )
    
    # train model
    print("[INFO] training network...")
    
    H = model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        batch_size=32,
        epochs=100,
        verbose=1
    )
    
    print("[INFO] serializing network...")
    
    model.save(args["model"])

    # evaluate model
    print("[INFO] evaliating network...")
    predictions = model.predict(
        testX,
        batch_size=32
    )
    
    print(
        classification_report(
            testY.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=["cat","dog","panda"]
        )
    )

    # plot performance
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(9, 6), dpi=150)
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.), 
        ncol=2, fancybox=True,
        shadow=True
        )
    plt.tight_layout()
    plt.savefig("../figures/shallownet_train.png")
    plt.close()

    



if __name__=="__main__":
    main()