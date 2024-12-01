import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(mp3_file, output_image):
    # Load the MP3 file
    y, sr = librosa.load(mp3_file, sr=None)  # sr=None to keep the original sample rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env,start_bpm=100,sr=sr)
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                   aggregate=None)
    print(f"This is the tempo {tempo[0]}")
    print(f"This is the dynamic tempo {dtempo}")

    cur_n_fft = 2048
    cur_hop_length = cur_n_fft // 4

    # Generate the spectrogram
    # Compute the Short-Time Fourier Transform (STFT) of the audio signal
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=cur_n_fft, hop_length=cur_hop_length)), ref=np.max)

    beat_per_second = 60/tempo[0]
    frames_per_second = int((beat_per_second * sr)//cur_hop_length)

    for i in range(len(D[0])//frames_per_second):

        d_sub = D[:, i*frames_per_second:(i+1)*frames_per_second]
        produceSpectrogramImage(d_sub, sr, output_image+""+f'{i}'+".png")
    '''
    print(type(frames_per_second))
    print(type(D))
    print(D.shape)

    print(type(d_sub))
    print(d_sub.shape)

    produceSpectrogramImage(d_sub, sr, output_image)
    '''


def produceSpectrogramImage(D, sr, output_image):
    print(len(D))
    print(len(D[0]))
    # Create a plot for the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', fmin=0, fmax=2000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Save the spectrogram as an image
    plt.savefig(output_image)
    plt.close()  # Close the plot to free memory

    print(f"Spectrogram saved as {output_image}")


if __name__ == "__main__":
    # Parses the argument taking in and adds Data prefix and mp3 suffix
    parses = argparse.ArgumentParser()
    parses.add_argument('file_path', type=str, help="Path to the input file")
    args = parses.parse_args()
    filepath = args.file_path
    input_file = "Data/" + args.file_path + ".mp3"
    parses.add_argument('file_path ')
# Example usage

    generate_spectrogram(input_file, 'Spectrogram/spectrogram')



'''import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import stft
from scipy.io import wavfile
import os

print("test")
def generate_spectrogram(audio_path, output_path):
    print("This ran")
    try:
        sampling_rate, data = wavfile.read(audio_path)
        if data.dtype == np.int16:  # For 16-bit PCM data
            data = data / 32768.0
        elif data.dtype == np.int32:  # For 32-bit PCM data
            data = data / 2147483648.0
        else:
            data = data / 1

        f, t, Zxx = stft(data, fs=sampling_rate, nperseg=1024)
        Zxx_dB = 20 * np.log10(np.abs(Zxx) + 1e-10)

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, Zxx_dB, shading='gouraud', cmap='magma')
        plt.colorbar(label='Amplitude (dB)')
        plt.title('Spectrogram')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.tight_layout()

        # Save the spectrogram
        plt.savefig(output_path)
        plt.close()
        print(f"Spectrogram saved to {output_path}")

    except Exception as e:

        print(f"Error: {e}")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Generate a spectrogram from an audio file.")
    parser.add_argument("audio_filename", type=str, help="Name of the audio file in the 'audio_files' directory.")
    args = parser.parse_args()

    print(f"Received audio file: {args.audio_filename}")




    # Define paths
    audio_path = args.audio_filename
    output_path = "spectrogram_scipy.png"
    generate_spectrogram(audio_path, output_path)
    # Check if file exists
'''