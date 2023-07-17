# 3DConvetNN

This is code written for the purpose of Prof. Camp's Wireless Systems Research

This 3-Dimensional Convolutional Neural Network takes in lidar data in the form of preprocessed voxels, along with
BTS2 data where we extract the column that contains the "Power of the receiver" values. These two parameters are passed into
the neural network and compared using the 'model.fit' function from the TensorFlow library to identify correlations in the data.

