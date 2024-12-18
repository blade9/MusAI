# import os
# import argparse
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np


# def generate_spectrogram(audio_file, output_dir):
#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Load the audio file
#     y, sr = librosa.load(audio_file, sr=None)  # sr=None to keep the original sample rate
#     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#     tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
#     print(f"This is the tempo: {tempo}")

#     cur_n_fft = 2048
#     cur_hop_length = cur_n_fft // 4

#     # Compute spectrogram
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=cur_n_fft, hop_length=cur_hop_length)), ref=np.max)
#     output_image = os.path.join(output_dir, 'spectrogram_original.png')
#     produceSpectrogramImage(D, sr, output_image)

#     # Compute frames per second based on tempo
#     beat_per_second = 60 / tempo  # beats per second
#     frames_per_second = int((beat_per_second * sr) // cur_hop_length)

#     # Create a directory to store spectrogram segments
#     tempogram_dir = "Spectrogram_Segments"
#     if not os.path.exists(tempogram_dir):
#         os.makedirs(tempogram_dir)

#     # Split the spectrogram into smaller segments based on the beats
#     for i in range(len(D[0]) // frames_per_second):
#         segment_start = frames_per_second * i
#         segment_end = frames_per_second * (i + 1)
        
#         # Extract the segment from the full spectrogram
#         spectrogram_segment = D[:, segment_start:segment_end]
        
#         # Produce and save each spectrogram segment
#         segment_output_path = os.path.join(tempogram_dir, f"segment_{i+1}.png")
#         produceSpectrogramImage(spectrogram_segment, sr, segment_output_path)
        

# def produceSpectrogramImage(D, sr, output_image):
#     # Plot and save the spectrogram
#     plt.figure(figsize=(10, 6))
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', fmin=0, fmax=2000)
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
#     plt.savefig(output_image)
#     plt.close()
#     print(f"Spectrogram saved as {output_image}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate spectrogram from an audio file.")
#     parser.add_argument('file_path', type=str, help="Path to the input audio file")
#     args = parser.parse_args()

#     input_file = args.file_path
#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"Input file not found: {input_file}")

#     song_name = os.path.splitext(os.path.basename(input_file))[0]
#     output_dir = os.path.join('Spectrogram', song_name)

#     generate_spectrogram(input_file, output_dir)




import os
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# from Models.lstmModel.NoteObject import NoteObject

def generate_spectrogram(audio_file, output_dir, time_signature=(4, 4)):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y, sr = librosa.load(audio_file, sr=None)  # sr=None to keep the original sample rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, start_bpm=100, sr=sr)
    print(f"This is the tempo: {tempo[0]}")


    cur_n_fft = 2048
    cur_hop_length = cur_n_fft // 4

    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=cur_n_fft, hop_length=cur_hop_length)), ref=np.max)
    output_image = os.path.join(output_dir, 'spectrogram_original.png')
    produceSpectrogramImage(D, sr, output_image)

    # Compute frames per second based on tempo
    # beat_per_second = 60 / tempo  # beats per second
    beat_per_second = 60 / tempo  # beats per second
    frames_per_second = int((beat_per_second * sr) // cur_hop_length)
    frames_per_second *= time_signature[0]

    # Create a directory to store spectrogram segments
    spectrogram_dir = "Spectrogram_Measures"
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)

    # Split the spectrogram into smaller segments based on the beats
    for i in range(len(D[0]) // frames_per_second):
        segment_start = frames_per_second * i
        segment_end = frames_per_second * (i + 1)
        
        # Extract the segment from the full spectrogram
        spectrogram_segment = D[:, segment_start:segment_end]
        segment_output_path = os.path.join(spectrogram_dir, f"measure_{i}")
        
        if not os.path.exists(segment_output_path):
            os.makedirs(segment_output_path)
        for j in range(time_signature[0]):
            beat_segment = spectrogram_segment[:, int(frames_per_second/4)*j:(j+1)*int(frames_per_second/4)]
            final_path = os.path.join(segment_output_path, f"spectrogram_{j}.png")
            txt_path = os.path.join(segment_output_path, f"spectrogram_{j}.txt")
            np.savetxt(txt_path, beat_segment, fmt="%0.6f")

            #This line is meant for testing purposes.  Uncomment this line out to visualize the individual
            #Spectrograms
            #produceSpectrogramImage(beat_segment, sr, final_path)

        
        
        # Produce and save each spectrogram segment
        '''
        segment_output_path = os.path.join(spectrogram_dir, f"spectrogram_{i+1}.png")
        produceSpectrogramImage(spectrogram_segment, sr, segment_output_path)
            spectrogram_segment = D[:, segment_start:segment_end]
        '''

def produceSpectrogramImage(D, sr, output_image):
    # Plot and save the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', fmin=0, fmax=2000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(output_image)
    plt.close()
    print(f"Spectrogram saved as {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrogram from an audio file.")
    parser.add_argument('file_path', type=str, help="Path to the input audio file")
    args = parser.parse_args()

    input_file = args.file_path
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    song_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join('Spectrogram', song_name)

    generate_spectrogram(input_file, output_dir)


