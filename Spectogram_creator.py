import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


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
    produceSpectrogramImage(D, sr, output_image + "" + 'original' + ".png")
    print(D.shape)
    np.set_printoptions(threshold=np.inf)
    for i in range(len(D[0])//frames_per_second):
        ons_splice = onset_env[frames_per_second*i:frames_per_second*(i+1)]
        produceTempogram(ons_splice, sr, cur_hop_length, i)
    '''
    for i in range(len(D[0])//frames_per_second):

        d_sub = D[:, i*frames_per_second:(i+1)*frames_per_second]
        produceSpectrogramImage(d_sub, sr, output_image+""+f'{i}'+".png")
    '''
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

def produceTempogram(onset_env, sr, hop_length, num):
    #onset_env = librosa.onset.onset_strength(y=audio_slice, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma')
    plt.colorbar(label='Strength')
    plt.title('Tempogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Tempo (BPM)')
    plt.tight_layout()
    plt.savefig("Tempogram/tempogram"+str(num))
    plt.close()

if __name__ == "__main__":
    # Parses the argument taking in and adds Data prefix and mp3 suffix
    parses = argparse.ArgumentParser()
    parses.add_argument('file_path', type=str, help="Path to the input file")
    args = parses.parse_args()
    filepath = args.file_path
    input_file = "Data/" + args.file_path + ".mp3"
    parses.add_argument('file_path ')
# Example usage

    output_dir = os.path.dirname("Spectrogram")
    print(output_dir)
    if not os.path.exists(output_dir):
        pass
    else:
        print("Successful")

    generate_spectrogram(input_file, 'Spectrogram/spectrogram')
