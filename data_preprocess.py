import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# List of audio file paths for different datasets
audio_dir_b = "data_meows/data_meows_b"
filenames_b= os.listdir(audio_dir_b)
audio_files_b = [os.path.join(audio_dir_b, filename) for filename in filenames_b if filename.endswith(".wav")]

audio_dir_i = "data_meows/data_meows_i"
filenames_i= os.listdir(audio_dir_i)
audio_files_i = [os.path.join(audio_dir_i, filename) for filename in filenames_i if filename.endswith(".wav")]


audio_dir_f = "data_meows/data_meows_f"
filenames_f= os.listdir(audio_dir_f)
audio_files_f = [os.path.join(audio_dir_f, filename) for filename in filenames_f if filename.endswith(".wav")]


# Initialize an empty list to store MFCCs
all_mfccs_b = []
all_mfccs_f = []
all_mfccs_i = []

# Loop through each audio file
for audio_file in audio_files_b:
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Compute MFCCs for the audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    # display the MFCCs for visualization
    # librosa.display.specshow(mfccs, x_axis='time')
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("MFCCs")
    # plt.show()
    
    # Append the MFCCs to the final array
    all_mfccs_b.append(mfccs)
    
for audio_file in audio_files_i:
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    all_mfccs_i.append(mfccs)
    
for audio_file in audio_files_f:
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    all_mfccs_f.append(mfccs)

#now all_mfccs is a list of 2D NumPy arrays where the dimensions are: (number of MFCC coefficients, number of time frames)


# Convert the list of MFCCs to a NumPy array
    #Pad all MFCCs to the same length
max_length = 0
for mfccs in all_mfccs_b:
    if mfccs.shape[1] > max_length:
        max_length = mfccs.shape[1]
for mfccs in all_mfccs_i:
    if mfccs.shape[1] > max_length:
        max_length = mfccs.shape[1]
for mfccs in all_mfccs_f:
    if mfccs.shape[1] > max_length:
        max_length = mfccs.shape[1]
        
for i in range(len(all_mfccs_b)):
    all_mfccs_b[i] = np.pad(all_mfccs_b[i], ((0, 0), (0, max_length - all_mfccs_b[i].shape[1])), 'constant')

for i in range(len(all_mfccs_i)):
    all_mfccs_i[i] = np.pad(all_mfccs_i[i], ((0, 0), (0, max_length - all_mfccs_i[i].shape[1])), 'constant')
    
for i in range(len(all_mfccs_f)):
    all_mfccs_f[i] = np.pad(all_mfccs_f[i], ((0, 0), (0, max_length - all_mfccs_f[i].shape[1])), 'constant')

final_mfccs_b = np.array(all_mfccs_b)
final_mfccs_i = np.array(all_mfccs_i)
final_mfccs_f = np.array(all_mfccs_f)

# final_mfccs is now a 3D NumPy array where the dimensions are: (440, 20, 173)
# (number of datasets, number of MFCC coefficients, number of time frames)


#save the MFCCs to a text file
#np.savetxt("mfccs_b.txt", final_mfccs_b.flatten()) #flatten the array to make it 2D before saving
#create lables for the data [1,0,0] for b, [0,1,0] for i, [0,0,1] for f
b_labels = []
i_labels = []
f_labels = []
for i in range(len(final_mfccs_b)):
    b_labels.append([1,0,0])
for i in range(len(final_mfccs_i)):
    i_labels.append([0,1,0])
for i in range(len(final_mfccs_f)):
    f_labels.append([0,0,1])


# Combine MFCCs and labels
combined_mfccs = np.concatenate((final_mfccs_b, final_mfccs_i, final_mfccs_f))
combined_labels = np.concatenate((b_labels, i_labels, f_labels))

#make labels array
combined_labels = np.array(combined_labels)
combined_mfcc= np.array(combined_mfccs)

#now we have a 3D NumPy array where the dimensions are: (number of datasets, number of MFCC coefficients, number of time frames) and a 1D NumPy array where the dimensions are: (number of datasets)

#shuffle and split the data to create training and testing datasets
from sklearn.model_selection import train_test_split

#split the data
x_train, x_test, y_train, y_test = train_test_split(combined_mfccs, combined_labels, test_size=0.1, random_state=42)

#normalize data, not necessary but makes it easier for the model to learn
import tensorflow as tf

x_train = tf.keras.utils.normalize(x_train, axis=1) #scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1) #scales data between 0 and 1

#save numpy array
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)