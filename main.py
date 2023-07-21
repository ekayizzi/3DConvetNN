# COMPLETED
# look at how to split the data using tensor flow
# Access the bts2 power column and store in data structure
# look at potential loss functions
# After training and evaluation, look how to refine the model in your notes ~ using Tensor Board
# prepare to receive npz files



# Sparse table library that tensor flow can read in

# Density of tensors
# You can also construct sparse tensors from dense tensors by using 'tf.sparse.from_dense',
# and convert them back to dense tensors by using 'tf.sparse.to_dense'

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import splitfolders

# Name of model so that we know for training/comparing models
# k = kernal size    m = max pooling size     f = number of functions
NAME = "3DConvetTest1-k7m2f128--" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Code for loading in npz files
# z = np.load(npzPath)
# print(z.files)   print(z['arr_0'])
# From there they are loaded in the variable and able to be used

X = ... # variable that will contain 3d voxel data which is contained in numpy arrays


# Code that reads in BTS2 data and transfer it to a numpy array
path = '/Users/eddiekayizzi/Downloads/BTS2.csv' # Add your file pathway to BTS2 here
data = pd.read_csv('r'+path)
signalPower = np.array(data['power of the reciver'])


# Variable containing a numpy array of signal power data # 1646 - 3291
y = signalPower

# Code used to split data, 80% into train data and 20% to testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Code that allows for data to be split from the directory into 3 output files
# How can I do this so that I only run this code?
# (make another python file in the same project? put the NN into a jupyter notebook? add a def function that outputs
# files
# once i run the code below it splits the data up, should I do that in another project? how do you have multiple open at the same time
directoryPath = '' # name of the file
splitfolders.ratio(directoryPath, output="output", seed=1337, ratio=(.8, 0.1,0.1))




# Predicted values are 3200, 40, 400, 1
# Waiting for Ryan for a more accurate number for each value
depth, height, width = 0






# This is the architecture of the neural network, will be tweaking kernal and max pooling size for optimization
# -- Convolutional aspect of the Neural Network --
# pool size (2,2,2)
# larger pool size leads to more down sampling - but risks losing spatial information
# smaller pool size helps retain more spatial details - but results in larger output sizes

# kernel size (3,3,3)
# known as the filter size of the receptive layer.
# refers to the dimension of the sliding window or filter used that moves across the input data

model = tf.keras.models.Sequential([

    # Takes in the input shape of the 3D Voxel Grid and filters / down samples to reduce noise
    tf.keras.layers.Conv3D(32, (7,7,7), activation='relu', input_shape=(depth, height, width, 1)),
    tf.keras.layers.MaxPooling3D(2,2,2),


    # Second layer of the Convolutional Aspect, to filter even more
    tf.keras.layers.Conv3D(64, (7,7,7), activation='relu'),
    tf.keras.layers.MaxPooling3D(2,2,2),

    # Beginning of Regular Neural Network
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')

])

# Compiles the model with an optimizer function and a loss function
model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Specify the log directory for TensorBoard
log_dir = 'logs/{}'.format(NAME)

# Create the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# To see active results go into terminal and run the /logs command



# For fitting the models together, (you possibly) need to put the npz data into a bigger array so that you can match them
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])


# model.evaluate? Shuffle/buffer variable present?

