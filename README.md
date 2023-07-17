# 3DConvetNN

This code is written for the purpose of Prof. Camp's Wireless Systems Research

This 3-Dimensional Convolutional Neural Network takes in lidar data in the form of preprocessed voxels, along with a
BTS2 file that contains the Lon Lat locations of various receivers with respect to a stationary transmitter. From this data we extract the column that contains the "Power of the receiver" that was found using the relationship from the Transmitter to the Receiver. The 3D Voxel Data is first passed through the 3D convolutional layer and filtered to extract the most important features so that the Neural Network has a better time processing this information and finding correlations. From there, these two parameters (Filtered 3D Voxel Data, Power of the Receiver) are passed into the neural network and compared using the 'model.fit' function from the TensorFlow library to identify correlations in the data.

