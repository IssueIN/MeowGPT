import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft


# Assuming you have functions for feature extraction
def load_data(audio_path):
    samplerate, data = wavfile.read(audio_path)
    
    return data

for emotion in ['B', 'I', 'C']:
    # Step 1: Preprocess the Cat Sound Dataset
    cat_sound_paths = os.listdir('data')  # Replace with your dataset
    cat_sounds = [load_data('data/' + path) for path in cat_sound_paths if path[0] == emotion]

    # Convert features to numpy array
    max_length = 0
    for cat_sound in cat_sounds:
        if cat_sound.shape[0] > max_length:
            max_length = cat_sound.shape[0]

    for i in range(len(cat_sounds)):
        for j in range(max_length - len(cat_sounds[i])):
            cat_sounds[i] = np.append(cat_sounds[i], 0)

    

    # Step 2: Train an Autoencoder
    cat_sounds = np.array(cat_sounds)
    input_shape = cat_sounds.shape
    print(input_shape)

    # Define Autoencoder model
    model = tf.keras.Sequential()
    model.add(Dense(input_shape[1], activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(input_shape[1], activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse')

    # Split dataset for training and testing
    X_train, X_test = train_test_split(cat_sounds, test_size=0.2, random_state=42)

    # Train Autoencoder
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    # Step 3: Generate Generic Cat Sound with certain emotion
    generic_cat_sound = model.predict(cat_sounds[0])
    

    plt.plot(generic_cat_sound)
    plt.show()

