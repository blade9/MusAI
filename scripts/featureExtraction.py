import math
import librosa
import numpy as np

def freq_to_note(frequency, base_frequency=440):
    """
    Determine the musical note name for a given frequency.

    Parameters:
        frequency (float): The frequency to analyze.
        base_frequency (float): The reference frequency for A4 (default is 440 Hz).

    Returns:
        tuple: The note name with the octave and the frequency in Hz.
    """
    # Define the note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Handle invalid frequency
    if frequency <= 0 or np.isnan(frequency):
        return None, None

    # Calculate the number of semitones between the given frequency and A4
    semitones = 12 * math.log2(frequency / base_frequency)

    # Round to the nearest semitone and determine the MIDI number
    midi_number = round(semitones) + 69  # MIDI number for A4 is 69

    # Determine the note name and octave
    note_name = note_names[midi_number % 12]
    octave = (midi_number // 12) - 1

    return f"{note_name}{octave}", frequency

# Load the audio file
y, sr = librosa.load('SeeYouAgain.mp3')  # Replace with your audio file

# Compute the Short-Time Fourier Transform (STFT) to get the spectrogram
D = np.abs(librosa.stft(y))

# Use librosa's piptrack to detect pitches and their magnitudes
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

# Create an empty list to store the detected notes, their times, and frequencies
note_changes = []

# Store the previous detected note
prev_note = None
start_time = None
prev_freq = None

# Loop through each time step in the spectrogram (each column in the pitch array)
for t in range(pitches.shape[1]):
    # Find the pitch with the highest magnitude at this time step
    index = magnitudes[:, t].argmax()
    frequency = pitches[index, t]

    # Convert the frequency to the corresponding musical note
    current_note, freq = freq_to_note(frequency)

    # If the note is valid (not None), check if it has changed
    if current_note and (prev_note is None or current_note != prev_note):
        if prev_note is not None and prev_freq is not None:
            duration = librosa.frames_to_time(t, sr=sr) - start_time
            note_changes.append((prev_note, prev_freq, start_time, duration))  # Store previous note, frequency, and duration
        prev_note = current_note  # Update the current note
        prev_freq = freq  # Update the current frequency
        start_time = librosa.frames_to_time(t, sr=sr)  # Record the start time of the new note

# Append the last note after the loop ends
if prev_note is not None and prev_freq is not None:
    duration = librosa.frames_to_time(t, sr=sr) - start_time
    note_changes.append((prev_note, prev_freq, start_time, duration))  # Store the last note and its duration

# Define the output text file
output_file = 'note_changes.txt'

# Write the detected notes, frequencies, and their durations to the file
with open(output_file, 'w') as f:
    f.write("Detected Notes (Note, Frequency in Hz, Start Time, Duration in seconds):\n")
    for note, freq, start, duration in note_changes:
        f.write(f"Note: {note}, Frequency: {freq:.2f} Hz, Start Time: {start:.2f} s, Duration: {duration:.2f} s\n")

print(f"Note changes have been written to {output_file}")
