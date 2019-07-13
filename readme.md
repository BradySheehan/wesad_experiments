# Stress Prediction with Limited Features

For this project, we are interested in using a RNN to predict when an individual
reaches a state of stress or increased stress.

Recently, we discovered the WESAD dataset (*/references*) that performed
experiments while measuring data from 12 subjects in which they indiced a 
stressful state using the TSST (Tier Social Stress Test). They recorded multiple
physiological parameters in addition to acceleration information during the
experiments. They attached two diferent devices to each individual; 
one wrist worn and one chest worn.

This project considers the chest worn device measurements.

## Features

This project is going to consider only 3 of the 6 sensor measurements from the
chest device. These measurements are accelerometer data, temperature data 
(skin temperature), and electrodermal activity (EDA) aka galvanic skin response
(GSR).

The features of interest for the ACC data are as follows.

1. mean for each axis; summed over all axes 
2. STD for each axis; summed over all axes 
2. Peak frequency for x
3. Peak frequency for Y
4. Peak frequency for z

The features of interest for the temperature sensor are:

1. Min value
2. max value
2. Dynamic Range
4. Mean 
5. STD

The features of interest for the EDA data are:

1. Min value
2. max value
2. Dynamic Range
4. Mean 
5. STD

## Project Structure and Development Process

### Environment

There are two assumed paths to run the notebooks and the python module:
- a path to the git project
- a path to the WESAD dataset
Be sure that these are assigned appropriately for yoru environment

### Dependencies

The project depends on multiple python libraries and packages. All of the code is 
written for Python3 and is using
- pandas
- matplotlib
- numpy
- keras
- tensorflow
- sklearn
- 



### File Structure
/
- demo.sh - driver setups, builds, trains, and tests model
- readme.md
- references/ - WESAD dataset and paper information
- src/
    - src/main - Python3 modules
        - DataManager.py
        - Demo.py
    - src/notebooks - jupyter ipython notebooks
        - exploring-the-dataset.ipynb
        - feature-exploration.ipynb
        - feature-exploration-continued.ipynb
        - model-training.ipynb
    - src/models - Directory for Keras model data files
        - model-5-epochs.json
        - model-5-epochs.h4
### Development Process

During the development process, trail and error experiments were performed
inside the notebooks. Anytime a new python package/library was needed, it was
imported and installed inside the notebook.

For each chunk of work that was completed, a functionwas added to the 
DataManager.py module. Along the way, there were some memory issues that could 
have been addressed better however all notebooks run as is.

## Results

## Future Work
In the future, we would like to refactor the DataManager to do incremental
data loading and feature calculations to improve performance. 
