#!/home/knielbo/virtenvs/cv/bin/python
"""
Train NeuralNetwork instance on mnist (subset of) as baseline
"""
from kartina.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
