# Speech Recognition Using Python
Speech recognition technology is one of the fast growing engineering technologies. It has a number of applications in different areas and provides potential benefits. About 20% of the world's people suffer from various disabilities, many of them blind or unable to use their hands effectively. Some people lose their hands in accidents. In those special cases, the speech recognition system provides them an important help, so that they can operate the computer through voice input and share information with people. The project is capable of recognizing speech and converting input audio into text.

## DATASET 
The dataset used in this project work has been taken from the Kaggle.com available at (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).It is a speech command datasheet issued by Tensorflow. The dataset has 65,000 long pronunciation of 30 short words by thousands of different people. 

## SYSTEM
To build this system, we have used various modules like Keras, TensorFlow, Librosa, Sounddevice, os, numpy, matplotlib , IPython.

## Model 
The model is based on deep learning using keras library consist of nine layers of neural network including input layer and output layer.
### First Input layer
It has given one argument “shape = (8000, 1)”. It is used to reshape input data into 1D column for 8000 elements.   
### Second Conv1d layer 1
In this layer the dimensionality of the output space (i.e. the number of output filters in the convolution) is 8 and kernel_size which specifying the length of the 1D convolution window is equal to 13.
### Third Conv1d layer 2
In this layer the dimensionality of the output space (i.e. the number of output filters in the convolution) is 16 and kernel_size which specifying the length of the 1D convolution window is equal to 11.
### Fourth Conv1d layer 3
In this layer the dimensionality of the output space (i.e. the number of output filters in the convolution) is 32 and kernel_size which specifying the length of the 1D convolution window is equal to 9.
### Fifth Conv1d layer 4
In this layer the dimensionality of the output space (i.e. the number of output filters in the convolution) is 64 and kernel_size which specifying the length of the 1D convolution window is equal to 7.
### Sixth Flatten layer
Flatten layer prepares a vector for the fully connected layers.
### Seventh Dense Layer 1
For this layer value of argument units = 256 which represent dimensionality of the output space.
### Eighth Dense Layer 2
For this layer value of argument units = 128 which represent dimensionality of the output space.
### Ninth Output Layer
A dense layer is used as output layer with units = length of labels which is dimensionality of the output space.

## Output


## Code
