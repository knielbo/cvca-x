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



if __name__=="__main__":
    main()