import os
import pretty_midi
import argparse
from NoteObject import NoteObject
from BeatObject import BeatObject


def process_midi_file(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    time_signature_data = midi_data.time_signature_changes

    tempo = midi_data.estimate_tempo()
    if len(time_signature_data) > 0:
        first_time = time_signature_data[0]
        numer = first_time.numerator
        denom = first_time.denominator
        time_signature = numer, denom
    else:
        time_signature = 4, 4

    bps = tempo/60
    all_beats = {}
    total_time = midi_data.get_end_time()
    total_beats = (total_time * tempo)/60
    measures = int(total_beats // time_signature[0])

    for i in range(measures):
        newBeat = BeatObject(i, [], tempo, time_signature[0], time_signature[1])
        all_beats[newBeat.getID()] = newBeat

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = note.start
            end_time = note.end
            duration_in_seconds = end_time - start_time

            duration_in_beats = bps * duration_in_seconds
            new_note = NoteObject(duration_in_beats, start_time, time_signature[0], time_signature[1])
            measure_index = int(start_time/(time_signature[0]/bps))
            all_beats[measure_index].addNote(new_note)

    output_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for i in range(len(all_beats)):
        # print(i)
        for my_note in all_beats[i].notes:
            start_time = my_note.starting_time
            duration = my_note.duration
            pitch = 60  # Middle C for simplicity
            velocity = 100  # Standard velocity
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time+duration)
            piano.notes.append(note)
            # print(my_note.getNoteType())

    output_midi.instruments.append(piano)
    output_dir = 'beat_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(all_beats)):
        print("Measure: ", i)
        for my_note in all_beats[i].notes:
            print("Note: ", my_note.getNoteType())

    song_name = os.path.splitext(os.path.basename(midi_path))[0]
    output_path = os.path.join(output_dir, f'{song_name}_output.mid')
    output_midi.write(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a MIDI file and output beats.')
    parser.add_argument('midi_path', type=str, help='Path to the input MIDI file')
    args = parser.parse_args()

    process_midi_file(args.midi_path)


