
import numpy as np
import librosa
import mido
from mido import MidiFile, MidiTrack, Message
from music21 import stream, note


# --- 1. Define the function to process audio and generate spectrogram ---
def process_audio_to_spectrogram(audio_path, sr=22050):
    """
    Convert audio file to spectrogram.
    :param audio_path: Path to the audio file.
    :param sr: Sample rate for audio.
    :return: Spectrogram representation of the audio.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return np.expand_dims(spectrogram, axis=-1)  # Add channel dimension for CNN


# --- 2. Define function to generate dummy ground truth ---
def generate_ground_truth(spectrogram, note_sequence=['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']):
    """
    Generate ground truth labels for a given spectrogram.
    :param spectrogram: The spectrogram array.
    :param note_sequence: A list of musical notes to cycle through for dummy labels.
    :return: A list of ground truth notes/chords for each spectrogram frame.
    """
    num_frames = spectrogram.shape[1]  # Number of time frames in the spectrogram
    num_notes = len(note_sequence)  # Length of the note sequence

    # Repeat the note sequence until we have enough labels for the number of frames
    ground_truth = (note_sequence * (num_frames // num_notes)) + note_sequence[:num_frames % num_notes]

    return ground_truth


# --- 3. Define function to convert note name to MIDI number ---
def note_name_to_number(note_name):
    """
    Convert a musical note name (e.g., 'C4', 'D#5') to its corresponding MIDI number.
    :param note_name: The note name as a string (e.g., 'C4', 'D#5').
    :return: The corresponding MIDI number.
    """
    note_mapping = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    # Extract the note and octave from the note name (e.g., 'C4' -> ('C', 4))
    note, octave = note_name[:-1], int(note_name[-1])
    
    # Calculate MIDI number: C4 is MIDI note 60, so adjust based on the octave
    midi_number = note_mapping[note] + (octave + 1) * 12
    return midi_number


# --- 4. Define function to create MIDI from predicted notes ---
def create_midi_from_notes(notes, output_file='output.mid', tempo=500000):
    """
    Convert a list of notes (e.g., ['C4', 'D4', 'E4', ...]) into a MIDI file.
    :param notes: List of notes in string format (e.g., ['C4', 'D4', ...]).
    :param output_file: The name of the output MIDI file.
    :param tempo: The tempo in microseconds per beat (500,000 is a tempo of 120 BPM).
    :return: None
    """
    # Create a new MIDI file and track
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Add a tempo message
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Add note-on and note-off messages for each note
    for note_str in notes:
        # Convert note string (e.g., 'C4') to MIDI number
        note_num = note_name_to_number(note_str)
        
        # Add note-on message (start of note)
        track.append(Message('note_on', note=note_num, velocity=64, time=0))
        
        # Add note-off message (end of note)
        track.append(Message('note_off', note=note_num, velocity=64, time=480))  # Duration of the note

    # Save the MIDI file
    midi.save(output_file)
    print(f"MIDI file saved as {output_file}")


# --- 5. Define function to create MusicXML from notes ---
def create_musicxml_from_notes(notes, output_file='output.xml'):
    """
    Convert a list of notes (e.g., ['C4', 'D4', 'E4', ...]) into a MusicXML file.
    :param notes: List of notes in string format (e.g., ['C4', 'D4', ...]).
    :param output_file: The name of the output MusicXML file.
    :return: None
    """
    # Create a stream (container) for the music
    score = stream.Score()

    # Create a part (like a track or instrument) for the score
    part = stream.Part()

    # Add the notes to the part
    for note_str in notes:
        # Convert note string (e.g., 'C4') to music21 note object
        new_note = note.Note(note_str)
        part.append(new_note)

    # Append the part to the score
    score.append(part)

    # Write the MusicXML file
    score.write('musicxml', fp=output_file)
    print(f"MusicXML file saved as {output_file}")


# --- 6. Main part of the program: Processing the audio and generating notes ---
audio_file = 'audio.mp3'  # Replace with your actual audio file path

# Process the audio file into a spectrogram
spectrogram = process_audio_to_spectrogram(audio_file)

# Generate dummy ground truth for testing (this is a placeholder, in practice, use actual ground truth)
ground_truth = generate_ground_truth(spectrogram)

# --- Reinforcement Learning part (Placeholder for your agent) ---
# Assuming that you have already trained the agent and it outputs notes as predicted from the spectrogram
# Here, for the sake of testing, let's just use the dummy ground truth as predicted notes
predicted_notes = ground_truth  # In practice, replace this with model output

# Choose between generating a MIDI or MusicXML file
create_midi_from_notes(predicted_notes, output_file='output.mid')  # For MIDI output
# or
create_musicxml_from_notes(predicted_notes, output_file='output.xml')  # For MusicXML output
