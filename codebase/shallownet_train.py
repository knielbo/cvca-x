#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of ShallowNet to animals3
    - illustrate save methods for trained models
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kartina.preprocessing import SimplePreprocessor
from kartina.preprocessing import SimplePreprocessor
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


if __name__=="__main__":
    main()