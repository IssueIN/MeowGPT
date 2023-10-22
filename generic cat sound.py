import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from sklearn import preprocessing
from pydub import AudioSegment
from pydub.playback import play
import aubio

# find mean frequency of cats
'''XB = np.array()

audio_dir_b = "data_meows/data_meows_b"
filenames_b= os.listdir(audio_dir_b)
audio_files_b = [os.path.join(audio_dir_b, filename) for filename in filenames_b if filename.endswith(".wav")]


for audio_file in audio_files_b:
    y, sr = librosa.load(audio_file)
    np.append(XB, y)

frequencies = np.array([fft(voice) for voice in XB])
sum_f = np.zeros(len(frequencies[0]))

for frequency in frequencies:
    sum_f = sum_f + frequency

indices = [i for i, x in enumerate(sum_f) if x == max(sum_f)]
if len(indices) == 1:
    dominant_f = indices[0]
else:
    dominant_f = int(np.mean(indices))

print(dominant_f)'''

for emotion in ['B', 'I', 'F']:

    X_train= np.load(emotion + 'x_train.npy')
    X_test= np.load(emotion + 'x_test.npy')

    input_shape = X_train.shape
    print(X_train[0].shape)

    # Define Autoencoder model
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(20, 173, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(173+40, activation='relu'),
    tf.keras.layers.Dense(173*10, activation='relu'),
    tf.keras.layers.Dense(173*20, activation='softmax'),
    tf.keras.layers.Reshape((20, 173))  
    ])


    model.compile(optimizer='adam', loss='mse')

    # Train Autoencoder
    model.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

    # Step 3: Generate Generic Cat Sound with certain emotion
    generic_cat_features = model.predict(X_test[:2])
    
    generic_cat_sound = librosa.feature.inverse.mfcc_to_audio(generic_cat_features[0])

    wavfile.write('generated' + emotion + '.wav', 22050, generic_cat_sound)



