import noisereduce as nr
import librosa
import soundfile as sf

def reduce_noise(input_path, output_path, noise_level=0.05):
    # Load the audio file
    audio_data, sample_rate = librosa.load(input_path, sr=None)

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(audio_data, sample_rate, prop_decrease=noise_level)

    # Save the result
    sf.write(output_path, reduced_noise, sample_rate)

# Example usage
input_path = 'generatedF.wav'
output_path = 'noise_reduced_generatedF.wav'
reduce_noise(input_path, output_path)

