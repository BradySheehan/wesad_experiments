# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:32:39 2019

@author: Brady Sheehan
"""

import os
import sys
# module = os.path.abspath('/home/learner/DLA_project/src/main')
module = os.path.abspath("C:/Users\\18145\\development\\wesad_experiments\\src\\main")
if module not in sys.path:
    sys.path.append(module)
import pip

import keras
import h5py
import ibmiotf
import tensorflow

import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

import sklearn
from  sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from DataManager import DataManager

manager = DataManager()

manager.load_all()

manager.compute_features()

manager.compute_features_stress0()

manager.train_model()

manager.test_model()