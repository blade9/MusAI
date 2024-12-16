from Models.lstmModel import MIDI_analyzer
import pretty_midi
import os, argparse

all_beats = MIDI_analyzer.process_midi_file('Data/babyslakh_16k/Track00001/MIDI/S00.mid')

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
    print("Measure: ", i)
    for my_note in all_beats[i].notes:
        print("Note: ", my_note.getNoteType())

song_name = 'pirates_midi_mine'
output_path = os.path.join(output_dir, f'{song_name}_output.mid')
output_midi.write(output_path)
