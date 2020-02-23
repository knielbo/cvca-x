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

