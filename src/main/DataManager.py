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
    # PATH = '/media/learner/6663-3462/WESAD/'
    ROOT_PATH = r'C:\WESAD'
    FILE_EXT = '.pkl'
    
    # IDs of the subjects
    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    
    # Dictionaries to store the two sets of data
    BASELINE_DATA = []
    STRESS_DATA = []
    
    BASELINE_FEATURES = []
    STRESS_FEATURES = []
    
    # Keys for measurements collected by the RespiBAN on the chest
    # minus the ones we don't want
    # RAW_SENSOR_VALUES = ['ACC','ECG','EDA','EMG','Resp','Temp']
    RAW_SENSOR_VALUES = ['ACC', 'EDA','Temp']
    
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
        print('Loading data from S'+ str(subject) + '\nPath=' + path)
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
        print("There are ",  values.size, " samples being considered.")
        num_features = values.size - window_size
        print("Computing ", num_features , " feature values with window size" \
              "of ", str(window_size) + "." )        
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
        print("There are ", len(values[:,1]), " samples being considered.")
        num_features = len(values[:,1]) - window_size
        print("Computing ", num_features , " feature values with window size" \
              "of ", str(window_size) + "." )
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
    
    def compute_features(self, subject, base_data=BASELINE_DATA, stress_data=STRESS_DATA, \
                         window_size=42000, window_shift=175):
        """
        returns:
        list: [base, stress], base = {'temp, 'ACC', 'EDA'}, stress = {'temp, 'ACC', 'EDA'}
        """
        
        #We are computing features for subjects 2 - 11, but indexing from 0
        index = subject - 2
        
        temp = base_data[index]['Temp']
        acc = base_data[index]['ACC']
        eda = base_data[index]['EDA']
        temp_stress = stress_data[index]['Temp']
        acc_stress = stress_data[index]['ACC']
        eda_stress = stress_data[index]['EDA']

        acc_features_base = self.get_features_for_acc(acc, window_size, window_shift)
        acc_features_stress = self.get_features_for_acc(acc_stress, window_size, window_shift)
        
        eda_features_stress = self.get_stats(eda, window_size, window_shift)
        eda_features_base = self.get_stats(eda_stress, window_size, window_shift)
        
        temp_features_base = self.get_stats(temp, window_size, window_shift)
        temp_features_stress = self.get_stats(temp_stress, window_size, window_shift)
        
        temp_base = [acc_features_base, eda_features_base, temp_features_base]
        temp_stress = [acc_features_stress, eda_features_stress, temp_features_stress]
        
        DataManager.BASELINE_FEATURES.append(temp_base)
        DataManager.STRESS_FEATURES.append(temp_stress)
        
        return [temp_base, temp_stress]
    
    def compute_all_features(self, subjects=SUBJECTS):
        for subject in subjects:
            print('Computing features for subject:', subject)
            self.compute_features(subject)
            
# TODO: Write a function that does 
# checks for if the data is not specified in the function being called
# then it throws an exception with a nice message.
            
#TODO: Write unit tests that assert the shape of the expected data.