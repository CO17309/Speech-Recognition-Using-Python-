# Import the libraries

import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")

path = "C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project"
os.listdir(path)

# Data Exploration and Visualization

# Visualization of Audio signal in time series domain

train_audio_path = "C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project\\train\\audio"
samples, sample_rate = librosa.load(train_audio_path+'\\yes\\0a7c2a8d_nohash_0.wav', sr = 16000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + "C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project\\train\\audio\\yes\\0a7c2a8d_nohash_0.wav")
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

# Sampling rate

ipd.Audio(samples, rate=sample_rate)

print(sample_rate)

# Resampling

samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)


# The number of recordings for each voice command:

labels=os.listdir(train_audio_path)

# Find count of each label and plot bar graph
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '\\'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
# Plot
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()


labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


# Duration of recordings

duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '\\'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '\\' + label + '\\' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))

# Preprocessing the audio waves
# Resampling
# Removing shorter commands of less than 1 second

train_audio_path = "C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project\\train\\audio"

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '\\'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '\\' + label + '\\' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)

# Convert the output labels to integer encoded:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

# Convert the integer encoded labels to a one-hot vector 
# Because it is a multi-classification problem

from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))

# Reshape the 2D array to 3D since the input to the conv1d must be a 3D array:

all_wave = np.array(all_wave).reshape(-1,8000,1)

# Split into train and validation set

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)

# Model building

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

# Defineing the loss function to be categorical cross-entropy since it is a multi-classification problem

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and to save the best model after every epoch:

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Train the model on a batch size of 32 and evaluate the performance on the holdout set

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

# Diagnostic plot

from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Loading the best model

from keras.models import load_model
model=load_model('best_model.hdf5')

# Defineing the function that predicts text for the given audio

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

# Make predictions on the validation data

import random
index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)

print("Text:",predict(samples))

# Prompts a user to record voice commands to test it on the model:

import sounddevice as sd
import soundfile as sf

while(1):
    option = input("Enter 'y' to say command of 1 second or 'n' to exit : ")
    if(option=='n'):
        break
    samplerate = 16000  
    duration = 1 # seconds
    filename = 'user_input.wav'
    print("START")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    print("END")
    sd.wait()
    sf.write(filename, mydata, samplerate)
    
    # Read the saved voice command and convert it to text
     
    os.listdir("C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project")
     
    filepath="C:\\Users\\AKHILESH THAPLIYAL\\Desktop\\DMA Lab Project"
    
    # Reading the voice commands
    samples, sample_rate = librosa.load(filepath + '\\' + 'user_input.wav', sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples,rate=8000)              
    
    # Converting voice commands to text
    predict(samples)
