#!/home/knielbo/virtenvs/cv/bin/python
"""
Driver script for application of LeNet to mnist

Usage
$ python
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

if __name__=="__main__":
    main()