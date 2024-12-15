import pretty_midi
from NoteObject import NoteObject
from Models.lstmModel.BeatObject import BeatObject


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


    for i in range(len(all_beats)):
        print(i)
        for my_note in all_beats[i].notes:
            print(my_note.getNoteType())



