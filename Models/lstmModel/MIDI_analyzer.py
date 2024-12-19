import pretty_midi
from Models.lstmModel.NoteObject import NoteObject
from Models.lstmModel.BeatObject import BeatObject


#This method takes in a midi file and outputs an beat_array of beats.  Each beat represets a measure in the song
# And each element in the sub array represents the array of notes it that particular beat
def process_midi_file(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    time_signature_data = midi_data.time_signature_changes

    tempo = round(midi_data.estimate_tempo())

    #Tempo calculations were not too good so just put the tempo of the actual midi here
    tempo = 145
    if len(time_signature_data) > 0:
        first_time = time_signature_data[0]
        numer = first_time.numerator
        denom = first_time.denominator
        time_signature = (numer, denom)
    else:
        time_signature = (4, 4)

    bps = tempo/60
    all_beats = {}
    total_time = midi_data.get_end_time()
    total_beats = (total_time * tempo)/60
    measures = int(total_beats // time_signature[0])
    measure_duration = round(time_signature[0] / bps, 4)

    for i in range(measures):
        newBeat = BeatObject(i, [], tempo, time_signature[0], time_signature[1])
        all_beats[newBeat.getID()] = newBeat

    list_start_time = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = round(note.start, 4)
            start_time = round(start_time - (measure_duration/time_signature[0])*2, 4)
            end_time = round(note.end, 4)
            end_time = round(end_time - (measure_duration/time_signature[0])*2, 4)
            duration_in_seconds = end_time - start_time
            duration_in_beats = round(bps * duration_in_seconds, 4)

            list_start_time.append(start_time)
            new_note = NoteObject(duration_in_beats, start_time, time_signature[0], time_signature[1])
            measure_index = int(start_time/measure_duration)
            all_beats[measure_index].addNote(new_note)

    for i in range(len(all_beats)):
        sum = 0
        for my_note in all_beats[i].notes:
            #print(NoteObject.NOTE_TYPES[my_note.getNoteType()])
            sum += NoteObject.NOTE_TYPES[my_note.getNoteType()]
            #print(my_note.getNoteType())
            #print(my_note.getStartTime())
            #print(my_note.getDuration())
            #print()
        #print(sum)
        #print()

    beat_calc = {}
    for i in range(len(all_beats)):
        beat_calc[i] = {}
        address = f"Measure", i
        for j in range(time_signature[0]):
            beat_calc[i][j] = []

        for my_note in all_beats[i].notes:
            mod_start = my_note.getStartTime() % measure_duration
            beatnum = mod_start // (measure_duration/time_signature[0])
            beat_calc[i][beatnum].append(my_note)

    return all_beats, beat_calc, tempo, time_signature




