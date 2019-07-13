# Stress Prediction with Limited Features

For this project, we are interested in using a RNN to predict when an individual
reaches a state of stress or increased stress. The goal of this project is to
become familiar with Keras for deep learning and the WESAD dataset.

Recently, we discovered the WESAD dataset (*/references*) that performed
experiments while measuring data from 12 subjects in which they induced a 
stressful state using the TSST (Tier Social Stress Test). They recorded multiple
physiological parameters in addition to acceleration information during the
experiments. They attached two diferent devices to each individual; 
one wrist worn and one chest worn and collected multiple modalities for each.

This project considers the chest worn device with a subset of the total number
of modalities.

## Directions

If you don't have the sd card loaded with all of the data on the NVIDIA Jetson TX2,
make sure that you point the DataManager.py and Notebooks to that data source.

After ensuring that the data is downloaded and referenced correctly, you can
run either the demo.sh script or the Demo.py script to test out the solution.

The demo will prepare, create, and evaluate an LSTM NN with one epoch and also
load and evalue an LSTM NN with 5 epochs. The demo prints the results from each
of these for your comparison.

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
- ipython and jupyter for jupyter notebook

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
        - model-5-epochs.h5
    
### Development Process

During the development process, trail and error experiments were performed
inside the notebooks. Anytime a new python package/library was needed, it was
imported and installed inside the notebook.

For each chunk of work that was completed, a functionwas added to the 
DataManager.py module. Along the way, there were some memory issues that could 
have been addressed better however all notebooks run as is.

## Results

Performance using the LSTM based network architecture with one hidden layer
has performed with an accuracy of ~ 97.7% on the validation set using 5 epochs.

Learning rate = 0.05
batch size = 2
With just one epoch, the model has results between ~80% and ~92% for accuracy
on the validation data. Each epoch of training takes approximately 70 seconds
without GPU acceleration. At 5 epochs, the model outperforms the WESAD quoated
results for both accuracy and F1 using less modalities and less features.


## Future Work

In the future, we would like to refactor the DataManager to do incremental
data loading and feature calculations to improve performance. 
