# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:38:18 2019

@author: Brady Sheehan
"""

import pickle
import numpy as np
import os

class Data:
    # Path to the SD Card
    PATH = '/media/learner/6663-3462/WESAD/'
    
    # Label values defined in the WESAD readme
    BASELINE = 1
    STRESS = 2
    
    # Dictionaries to store the two sets of data
    baseline_data = []
    stress_data = []
    
    # Keys for measurements collected by the RespiBAN on the chest
    RAW_SENSOR_VALUES = ['ACC','ECG','EDA','EMG','Resp','Temp']
    
    
    def __init__(self, ignore_empatica=True):
        # self.subject = subject
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
        path = Data.PATH + 'S' + str(subject) + '/S' + str(subject) + '.pkl'
        print('Loading from\nPath=' + path)
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
    
    def extract_and_reform(self, data, subject):
        """ 
        Extracts and shapes the data from the pkl file
        for the provided subject.
        
        Parameters:
        data (dict): as loaded from the pickle file
        
        Returns: Baseline and stress data
        dict: {{'ECG': [###, ..], ..}, 
               {'ECG': [###, ..], ..} }
        """
        
        if self.ignore_empatica:
            del data['signal']['wrist']
        
        baseline_indices = np.nonzero(data['label']==Data.BASELINE)[0]   
        stress_indices = np.nonzero(data['label']==Data.STRESS)[0]
        base = dict()
        stress = dict()
        
        for value in Data.RAW_SENSOR_VALUES:
            base[value] = data['signal']['chest'][value][baseline_indices]
            stress[value] = data['signal']['chest'][value][stress_indices]
        
        Data.baseline_data.append(base)
        Data.stress_data.append(stress)
        
        return base, stress


class Subject:
    
    def __init__(self, subject):
        # self.subject = subject
        self.subject = subject