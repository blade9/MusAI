import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram(mp3_file, output_image):
    # Load the MP3 file
    y, sr = librosa.load(mp3_file, sr=None)  # sr=None to keep the original sample rate

    cur_n_fft = 4096
    cur_hop_length = 512
    # Generate the spectrogram
    # Compute the Short-Time Fourier Transform (STFT) of the audio signal
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft = cur_n_fft, hop_length = cur_hop_length)), ref=np.max)

    # Create a plot for the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    # Save the spectrogram as an image
    plt.savefig(output_image)
    plt.close()  # Close the plot to free memory

    print(f"Spectrogram saved as {output_image}")

if __name__ == "__main__":

# Example usage
    generate_spectrogram('See_you_again_song.mp3', 'spectrogram.png')



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