import pretty_midi
from collections import defaultdict

midi_file_path = '../data/extracted_stems/MIDI/melody0.mid'
output_path = '../outputs'

def group_pitches_by_start_time(midi_file_path, time_tolerance=0.01):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
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

def create_midi_from_group(grouped_notes, output_dir):
    for start_time, notes in grouped_notes.items():
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

        # Save to a MIDI file
        output_file = f"{output_dir}/group_{start_time:.3f}.mid"
        midi.write(output_file)