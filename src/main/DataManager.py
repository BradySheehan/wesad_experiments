# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:38:18 2019

@author: Brady Sheehan
"""

import pickle
import numpy as np
import os
import datetime
import tensorflow as tf
from pathlib import Path

import sklearn
from  sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.models import load_model

class DataManager:
    # Path to the WESAD dataset
    ROOT_PATH = '/media/learner/6663-3462/WESAD/'
    #ROOT_PATH = r'C:\WESAD'
    
    # pickle file extension for importing
    FILE_EXT = '.pkl'

    # Directory in project structure where model files are stored
    MODELS_DIR = os.path.join(Path().absolute().parent, 'models')

    # IDs of the subjects
    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    
    FEATURE_KEYS =     ['max',  'min', 'mean', 'range', 'std']
    FEATURE_ACC_KEYS = ['maxx', 'maxy', 'maxz', 'mean', 'std']

    # Keys for measurements collected by the RespiBAN on the chest
    # minus the ones we don't want
    # RAW_SENSOR_VALUES = ['ACC','ECG','EDA','EMG','Resp','Temp']
    RAW_SENSOR_VALUES = ['ACC', 'EDA','Temp']
    
    FEATURES = {'a_mean': [], 'a_std': [], 'a_maxx': [], 'a_maxy': [], 'a_maxz': [],\
                'e_max': [],  'e_min': [], 'e_mean': [], 'e_range': [], 'e_std': [], \
                't_max': [],  't_min': [], 't_mean': [], 't_range': [], 't_std': [] }
    STRESS_FEATURES = {'a_mean': [], 'a_std': [], 'a_maxx': [], 'a_maxy': [], 'a_maxz': [],\
                'e_max': [],  'e_min': [], 'e_mean': [], 'e_range': [], 'e_std': [], \
                't_max': [],  't_min': [], 't_mean': [], 't_range': [], 't_std': [] }
    
    # Dictionaries to store the two sets of data
    BASELINE_DATA = []
    STRESS_DATA = []
    
    # the file name for the last created model
    last_saved=''
    
    def __init__(self, ignore_empatica=True, ignore_additional_signals=True):
        # denotes that we will be excluding the empatica data 
        # after loading those measurements
        self.ignore_empatica = ignore_empatica

    def get_subject_path(self, subject):
        """ 
        Parameters:
        subject (int): id of the subject
        
        Returns:
        str: path to the pickle file for the given subject number
             iff the path exists 
        """
        
        # subjects path looks like data_set + '<subject>/<subject>.pkl'
        path = os.path.join(DataManager.ROOT_PATH, 'S'+ str(subject), 'S' + str(subject) + DataManager.FILE_EXT)
        print('Loading data for S'+ str(subject))
        #print('Path=' + path)
        if os.path.isfile(path):
            return path
        else:
            print(path)
            raise Exception('Invalid subject: ' + str(subject))

    def load(self, subject):
        """ 
        Loads and saves the data from the pkl file for the provided subject
        
        Parameters:
        subject (int): id of the subject
        
        Returns: Baseline and stress data
        dict: {{'EDA': [###, ..], ..}, 
               {'EDA': [###, ..], ..} }
        """
       
        # change the encoding because the data appears to have been
        # pickled with py2 and we are in py3
        with open(self.get_subject_path(subject), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            return self.extract_and_reform(data, subject)
    
    def load_all(self, subjects=SUBJECTS):
        for subject in subjects:
            self.load(subject)
                
    
    def extract_and_reform(self, data, subject):
        """ 
        Extracts and shapes the data from the pkl file
        for the provided subject.
        
        Parameters:
        data (dict): as loaded from the pickle file
        
        Returns: Baseline and stress data
        dict: {{'EDA': [###, ..], ..}, 
               {'EDA': [###, ..], ..} }
        """
                
        if self.ignore_empatica:
            del data['signal']['wrist']
        
        baseline_indices = np.nonzero(data['label']==DataManager.BASELINE)[0]   
        stress_indices = np.nonzero(data['label']==DataManager.STRESS)[0]
        base = dict()
        stress = dict()
        
        for value in DataManager.RAW_SENSOR_VALUES: 
            base[value] = data['signal']['chest'][value][baseline_indices]
            stress[value] = data['signal']['chest'][value][stress_indices]
        
        DataManager.BASELINE_DATA.append(base)
        DataManager.STRESS_DATA.append(stress)
        
        return base, stress
    
    def get_stats(self, values, window_size=42000, window_shift=175):
        """ 
        Calculates basic statistics including max, min, mean, and std
        for the given data
        
        Parameters:
        values (numpy.ndarray): list of numeric sensor values
        
        Returns: 
        dict: 
        """
#        print("There are ",  values.size, " samples being considered.")
        num_features = values.size - window_size
#        print("Computing ", num_features , " feature values with window size" \
#              "of ", str(window_size) + "." )        
        max_tmp = []
        min_tmp = []
        mean_tmp = []
        dynamic_range_tmp = []
        std_tmp = []
        for i in range(0, num_features, window_shift):
            window = values[i:window_size + i]
            max_tmp.append(np.amax(window))
            min_tmp.append(np.amin(window))
            mean_tmp.append(np.mean(window))
            dynamic_range_tmp.append(max_tmp[-1] - min_tmp[-1])
            std_tmp.append(np.std(window))

        features = {}
        features['max'] = max_tmp
        features['min'] = min_tmp
        features['mean'] = mean_tmp
        features['range'] = dynamic_range_tmp
        features['std'] = std_tmp
        return features

    def get_features_for_acc(self, values, window_size=42000, window_shift=175):
        """ 
        Calculates statistics including mean and std
        for the given data and the peak frequency per axis and the
        body acceleration component (RMS)
        
        Parameters:
        values (numpy.ndarray): list of numeric sensor values [x, y, z]
        
        Returns: 
        dict: 
        """
        #print("There are ", len(values[:,1]), " samples being considered.")
        num_features = len(values[:,1]) - window_size
        #print("Computing ", num_features , " feature values with window size" \
        #              "of ", str(window_size) + "." )
        maxx_tmp = []
        maxy_tmp = []
        maxz_tmp = []
        mean_tmp = []
        std_tmp = []        
        for i in range(0, num_features, window_shift):
            windowx = values[i:window_size + i, 0]
            windowy = values[i:window_size + i, 1]
            windowz = values[i:window_size + i, 2]
                        
            meanx = np.mean(windowx)
            meany = np.mean(windowy)
            meanz = np.mean(windowz)
            mean_tmp.append( (meanx + meany + meanz) )

            stdx = np.std(windowx)
            stdy = np.std(windowy)
            stdz = np.std(windowz)
            std_tmp.append( (stdx + stdy + stdz) )
            
            maxx_tmp.append(np.amax(windowx))
            maxy_tmp.append(np.amax(windowy))
            maxz_tmp.append(np.amax(windowz))

        features = {}
        features['mean'] = mean_tmp
        features['std'] =  std_tmp
        features['maxx'] = maxx_tmp
        features['maxy'] = maxy_tmp
        features['maxz'] = maxz_tmp
        
        return features
    
    def compute_features(self, subjects=SUBJECTS, data=BASELINE_DATA, window_size=42000, window_shift=175):
        keys = list(DataManager.FEATURES.keys())
        print('Computing features..')
        for subject in subjects:
            print("\tsubject:", subject)
            index = subject - 2
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in DataManager.FEATURE_ACC_KEYS:
                #print('computed ', len(acc[feature]), 'windows for acc ', feature)
                DataManager.FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
                #print('computed ', len(eda[feature]), 'windows for eda ', feature)
                DataManager.FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
                #print('computed ', len(temp[feature]), 'windows for temp ', feature)
                DataManager.FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
            
        return DataManager.FEATURES

    def compute_features_stress(self, subjects=SUBJECTS, data=STRESS_DATA, window_size=42000, window_shift=175):
        keys = list(DataManager.STRESS_FEATURES.keys())
        print('conputing features..')    
        for subject in subjects:
            print("\tsubject:", subject)
            index = subject - 2
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in DataManager.FEATURE_ACC_KEYS:
                #print('computed ', len(acc[feature]), 'windows for acc ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
                #print('computed ', len(eda[feature]), 'windows for eda ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
                #print('computed ', len(temp[feature]), 'windows for temp ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
        return DataManager.STRESS_FEATURES

    def get_train_and_test_data(self):
        X1 = []
        X2 = []
        for i in range(0, len(DataManager.FEATURES['a_mean'])):
            X1.append([DataManager.FEATURES['a_mean'][i], DataManager.FEATURES['a_std'][i],\
                       DataManager.FEATURES['a_maxx'][i], DataManager.FEATURES['a_maxy'][i],\
                       DataManager.FEATURES['a_maxz'][i], DataManager.FEATURES['e_max'][i],\
                       DataManager.FEATURES['e_min'][i],  DataManager.FEATURES['e_mean'][i],\
                       DataManager.FEATURES['e_range'][i],DataManager.FEATURES['e_std'][i],\
                       DataManager.FEATURES['t_max'][i],  DataManager.FEATURES['t_min'][i],\
                       DataManager.FEATURES['t_mean'][i], DataManager.FEATURES['t_range'][i],\
                       DataManager.FEATURES['t_std'][i]])
        #print(np.shape(X1))
        
        for i in range(0,  len(DataManager.STRESS_FEATURES['a_mean'])):
            X2.append([DataManager.STRESS_FEATURES['a_mean'][i], DataManager.STRESS_FEATURES['a_std'][i],\
                       DataManager.STRESS_FEATURES['a_maxx'][i], DataManager.STRESS_FEATURES['a_maxy'][i],\
                       DataManager.STRESS_FEATURES['a_maxz'][i], DataManager.STRESS_FEATURES['e_max'][i],\
                       DataManager.STRESS_FEATURES['e_min'][i], DataManager.STRESS_FEATURES['e_mean'][i],\
                       DataManager.STRESS_FEATURES['e_range'][i], DataManager.STRESS_FEATURES['e_std'][i],\
                       DataManager.STRESS_FEATURES['t_max'][i], DataManager.STRESS_FEATURES['t_min'][i],\
                       DataManager.STRESS_FEATURES['t_mean'][i], DataManager.STRESS_FEATURES['t_range'][i],\
                       DataManager.STRESS_FEATURES['t_std'][i]])                
        #print(np.shape(X2))
        
        # initialize zero for base and 1 for stress
        y1 = [0] * len(X1)
        y2 = [1] * len(X2)
        # Now we need to concat the data between baseline and stress so that 
        # we can split it into training and test sets    
        X = np.concatenate((X1, X2), axis=0)
        #print(np.shape(X))
        
        y = np.concatenate((y1,y2), axis=0)
        #print(np.shape(y))
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.25, random_state=42)
        return (X_train, X_test, y_train, y_test)

    def normalize(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data)

    def scale_data(self, X_train, X_test, y_train, y_test):
        print("Scaling the data...")
        (X_train, X_test, y_train, y_test) = self.get_train_and_test_data()
        X_train = self.normalize(X_train)
        X_test = self.normalize(X_test)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        return (X_train, X_test, y_train, y_test)

    def build_model(self):
        num_neurons = 15
        num_features = 15
        
        print('Building the LSTM NN...')

        model = Sequential()
        model.add(LSTM(num_neurons, input_shape=(1, num_features), return_sequences=True))
        model.add(LSTM(num_neurons, input_shape=(1, num_features), return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        
        model = self.configure_learning(model)
        print(model.summary())
        return model

    def configure_learning(self, model):
        opt = SGD(lr=0.05)
        model.compile(loss='binary_crossentropy', optimizer=opt,\
                      metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, X_test, y_train, y_test,\
                    batch_size=2, epochs=5):
        print('Training network...')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, y_test))
        #print("inputs: " , model.input_shape)
        #print("outputs: ", model.output_shape)
        #print("actual inputs: ", np.shape(X_train))
        #print("actual outputs: ", np.shape(y_train))
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
        return (model, score, acc)

    def load_model(self, file_name=last_saved):
        # Load the model of interest
        print("Loading model:", file_name)
        file = (os.path.join(DataManager.MODELS_DIR, file_name))
        model_from_disc = load_model(file)
        return model_from_disc

    def save_model(self, model):
        now = datetime.datetime.now()
        # Make sure the datetime str has no special characters and no spaces
        DataManager.last_saved = str("model-" + \
                                     str(now.replace(microsecond=0)) +\
                                     ".h5").replace(" ", "").replace(":", "_")
        model.save(os.path.join(DataManager.MODELS_DIR, DataManager.last_saved))
        print("Saved model to disc:",\
              DataManager.last_saved)
        
    def get_model_results(self, model, X_train, X_test, y_train, y_test,\
                          batch_size=2):
        print('batch_size = ', batch_size)
        print('Model results from model.evaluate() test data')
        score, acc = model.evaluate(X_test , y_test, batch_size=batch_size)
        print('score:', score, 'accuracy:', acc)
        
        y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
        y_pred[y_pred>0.5] = 1 
        y_pred[y_pred<=0.5] = 0 
        print("_________________________________________________________________")
        print('\nClassification report from model.predict with test data')
        print(classification_report(y_test, y_pred))
        print("_________________________________________________________________")
        print('\nConfusion matrix from model.predict with test data')
        print(confusion_matrix(y_test, y_pred))
        print("_________________________________________________________________")

    def create_network(self, epochs=5, batch_size=2):
        (X_train, X_test, y_train, y_test) = self.get_train_and_test_data()
        (X_train, X_test, y_train, y_test) = \
            self.scale_data(X_train, X_test, y_train, y_test)
        model = self.build_model()
        (model, score, acc) = self.train_model(model, X_train, X_test, y_train, y_test,\
                                 batch_size, epochs)
        self.save_model(model)
        
        #self.get_model_results(model, X_train, X_test, y_train, y_test)
        return (model, X_train, X_test, y_train, y_test)

# TODO: Write a function that does 
# checks for if the data is not specified in the function being called
# then it throws an exception with a nice message.
            
#TODO: Write unit tests that assert the shape of the expected data.
        
#TODO: Break out some of these functions about the model specifically 
# into a seperate class from the functions about the data.
