import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_file_path = "meow_gpt_data/dataset/B_ANI01_MC_FN_SIM01_101.wav"
y, sr = librosa.load(audio_file_path)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# display the MFCCs for visualization
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar(format="%+2.0f dB")
plt.title("MFCCs")
plt.show()