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

from DataManager import DataManager

manager = DataManager()

manager.load_all()
manager.compute_features()
manager.compute_features_stress()

batch_size = 2
epochs = 1

# compute for one epoch
(model, X_train, X_test, y_train, y_test) = \
    manager.create_network(epochs, batch_size)

# then load a previously computed 5 epoch model and display the results
model_5_epochs = manager.load_model()
manager.get_model_results(model_5_epochs, X_train, X_test, y_train, y_test )

