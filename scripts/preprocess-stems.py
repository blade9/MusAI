from collections import defaultdict
import numpy as np
import pretty_midi
import pprint

midi_file_path = '../data/extracted_stems/MIDI/melody0.mid'
NUM_PITCHES = 128

def group_pitches_by_onset(melody_path, time_tolerance=0.01):
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

def midi_to_spectrogram(midi):


if __name__ == '__main__':
    grouped_notes = group_pitches_by_onset(midi_file_path)
    unique_groups = remove_redundant_groups(grouped_notes)
    for start_time, notes in unique_groups.items():
        midi = create_midi_from_group(start_time, notes)
        label = get_midi_label(midi)
        break
    print(label)