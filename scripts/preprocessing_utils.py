from collections import defaultdict
from midi2audio import FluidSynth

import matplotlib.pyplot as plt
import librosa
import numpy as np
import pretty_midi
import pprint

NUM_PITCHES = 128

def group_pitches_by_onset(melody_path, time_tolerance=0.01):
    if isinstance(melody_path, bytes):
        melody_path = melody_path.decode('utf-8')
    midi_data = pretty_midi.PrettyMIDI(melody_path)
    grouped_notes = defaultdict(list)

    # Iterate through all notes in all instruments
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Round start time to the nearest multiple of time_tolerance
            group_time = round(note.start / time_tolerance) * time_tolerance
            grouped_notes[group_time].append(note)

    return grouped_notes

def remove_redundant_groups(grouped_notes):
    unique_groups = {}
    seen_pitches = set()
    for time, notes in grouped_notes.items():
        pitch_set = tuple(sorted(note.pitch for note in notes))
        if pitch_set not in seen_pitches:
            seen_pitches.add(pitch_set)
            unique_groups[time] = notes
    return unique_groups

def create_midi_from_group(start_time, notes):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Add notes to the instrument
    for note in notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=0,  # Reset start times relative to the group
            end=1 # Adjust end time to be relative
        ))

    midi.instruments.append(instrument)
    return midi

def get_midi_label(midi):
    label = np.zeros(NUM_PITCHES, dtype=int)
    for instrument in midi.instruments:
        for note in instrument.notes:
            label[note.pitch] = 1
    return label

def midi_to_spectrogram(midi, sample_rate=44100, n_fft=2048, hop_length=512):
    audio_wave = midi.fluidsynth(fs=sample_rate)
    spectrogram = librosa.stft(audio_wave, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    return spectrogram_db

def plot_spectrogram(spectrogram, sample_rate=44100, hop_length=512):
    """
    Plot the spectrogram for visualization.
    """
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()