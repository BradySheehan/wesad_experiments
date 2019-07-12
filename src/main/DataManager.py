# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:38:18 2019

@author: Brady Sheehan
"""

import pickle
import numpy as np
import os
#import statistics as stat
#import matplotlib.pyplot as plt
#import pandas as pd

class DataManager:
    # Path to the SD Card
    ROOT_PATH = '/media/learner/6663-3462/WESAD/'
    # ROOT_PATH = r'C:\WESAD'
    FILE_EXT = '.pkl'
    
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
        print('Loading data from S'+ str(subject) + '\n\tPath=' + path)
        if os.path.isfile(path):
            return path
        else:
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
#        print("There are ", len(values[:,1]), " samples being considered.")
        num_features = len(values[:,1]) - window_size
#        print("Computing ", num_features , " feature values with window size" \
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
        print('conputing features..')
        for subject in subjects:
            print("Considering subject ", subject)
            index = subject - 2
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in DataManager.FEATURE_ACC_KEYS:
#                print('computed ', len(acc[feature]), 'windows for acc ', feature)
                DataManager.FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
#                print('computed ', len(eda[feature]), 'windows for eda ', feature)
                DataManager.FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
#                print('computed ', len(temp[feature]), 'windows for temp ', feature)
                DataManager.FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
            
        return DataManager.FEATURES

    def compute_features_stress(self, subjects=SUBJECTS, data=STRESS_DATA, window_size=42000, window_shift=175):
        keys = list(DataManager.STRESS_FEATURES.keys())
        print('conputing features..')    
        for subject in subjects:
            print("Considering subject ", subject)
            index = subject - 2
            key_index = 0
            
            acc = self.get_features_for_acc(data[index]['ACC'], window_size, window_shift)
            for feature in DataManager.FEATURE_ACC_KEYS:
#                print('computed ', len(acc[feature]), 'windows for acc ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(acc[feature])
                key_index = key_index + 1
            
            eda = self.get_stats(data[index]['EDA'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
#                print('computed ', len(eda[feature]), 'windows for eda ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(eda[feature])
                key_index = key_index + 1

            temp = self.get_stats(data[index]['Temp'], window_size, window_shift)
            for feature in DataManager.FEATURE_KEYS:
#                print('computed ', len(temp[feature]), 'windows for temp ', feature)
                DataManager.STRESS_FEATURES[keys[key_index]].extend(temp[feature])
                key_index = key_index + 1
        return DataManager.STRESS_FEATURES

# TODO: Write a function that does 
# checks for if the data is not specified in the function being called
# then it throws an exception with a nice message.
            
#TODO: Write unit tests that assert the shape of the expected data.