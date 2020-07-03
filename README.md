# Speech Recognition Using Python
Speech recognition technology is one of the fast growing engineering technologies. It has a number of applications in different areas and provides potential benefits. About 20% of the world's people suffer from various disabilities, many of them blind or unable to use their hands effectively. Some people lose their hands in accidents. In those special cases, the speech recognition system provides them an important help, so that they can operate the computer through voice input and share information with people. The project is capable of recognizing speech and converting input audio into text.

## DATASET 
The dataset used in this project work has been taken from the Kaggle.com available at ( https://www.kaggle.com/c/tensorflow-speech-recognition-challenge ). It is a speech command datasheet issued by Tensorflow. The dataset has 65,000 long pronunciation of 30 short words by thousands of different people. 

## SYSTEM
To build this system, we have used various modules like Keras, TensorFlow, Librosa, Sounddevice, os, numpy, matplotlib , IPython. It has two stages. The first stage include training and testing of model based on deep learning using keras library consist of nine layers of neural network including input layer and output layer. The second stage mainly includes taking one second audio input from user and predicting its value. 

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

### Output 1

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/Output%201.png

In this project, we have used lots of voice commands and, this is the output, which displays the number of recordings of each command.

### Output 2

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/Output%202.png

Here, this is the output of the code where we have checked the duration of the all audio commands. Through this histogram it is clear that maximum voice commands are of one second.

### Output 3

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/Output%203.png

Data Exploration and Visualization helps us to understand the data as well as the preprocessing steps in a better way. This is the output of the visualization of Audio signals in time domain series.


### Output 4

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/Output%205.png

Next, we have the output of the code where we have trained our model and tested it. Model predict the voice\audio command with the accuracy of 0.8765. In this graph ,Blue line shows the training part and orange line shows the testing. 

### Output 5

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/Output%205.png

This is the final output of our code ,where it asks from the user to speak for one second.
At start user speaks and at end command user stops ,and then the model predicts the voice and convert that audio into text.

## Code

https://github.com/CO17309/Speech-Recognition-Using-Python-/blob/master/CO17308%20CO17309%20Project%20Speech%20Recognition.py

In this program we have trained speech recognition model which can few basic commands from speech to text in real time. It can recognize commands such as "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", etc. This program has been executed in two stages. The first stage include training and testing of model based on deep learning using keras library consist of nine layers of neural network including input layer and output layer. The second stage mainly includes taking one second audio input from user and predicting its value. 
