from Models.lstmModel import MIDI_analyzer
from Models.lstmModel.NoteObject import NoteObject
import pretty_midi
import os, argparse

all_beats, beat_times = MIDI_analyzer.process_midi_file('Data/babyslakh_16k/Track00001/MIDI/S00.mid')

for i in beat_times:
    print(i)
    print("This is our measure")
    for j in beat_times[i]:
        print("This is our beat")
        print(j)
        for my_note in beat_times[i][j]:
            print(my_note.getNoteType())

data_in_input = []
for i in beat_times:
    spec_file_path = f"Spectrogra_Measures/measures_{i+1}"
    for j in beat_times[i]:
        spec_file_new = spec_file_path + f"/spectrogram_{j}"
        note_data = []
        for my_note in beat_times[i][j]:
            note_data.append(my_note.getData())
        txtline = spec_file_new + ", " + str(note_data)
        data_in_input.append(txtline)


with open("input.txt", "w") as file:
    for line in data_in_input:
        file.write(line+"\n")



# Go through the all_beats array by measure
# Mod the starting time by the time of the measure taking
# Group the starting based on what division based


output_midi = pretty_midi.PrettyMIDI(initial_tempo=200)
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)
bps = 200/60
output_midi.time_signature_changes.append(pretty_midi.TimeSignature(6, 8, time=0.0))


for i in range((len(all_beats))):
    # print(i)
    for my_note in all_beats[i].notes:
        start_time = my_note.getStartTime()
        duration = my_note.getDuration()
        pitch = 60  # Middle C for simplicity
        velocity = 100  # Standard velocity
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time+(duration/bps))
        piano.notes.append(note)
        # print(my_note.getNoteType())

output_midi.instruments.append(piano)
output_dir = 'beat_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(all_beats)):
    #print("Measure: ", i)
    for my_note in all_beats[i].notes:
        #print("Note: ", my_note.getNoteType())
        pass

song_name = 'pirates_midi_mine'
output_path = os.path.join(output_dir, f'{song_name}_output.mid')
output_midi.write(output_path)
