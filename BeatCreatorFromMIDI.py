import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import Models.lstmModel.LSTM_model
from Models.lstmModel import MIDI_analyzer
from Models.lstmModel.NoteObject import NoteObject
from Models.lstmModel.BeatObject import BeatObject
from scripts.splitspectrogram import *
from Models.lstmModel import LSTM_model

import pretty_midi
import os, argparse

parser = argparse.ArgumentParser(description="Process a MIDI file and generate a MIDI output.")
parser.add_argument("midi_file", type=str, help="Path to the input MIDI file")
parser.add_argument("wav_file", type=str, help="Path to the original WAV file of the song")

args = parser.parse_args()
midi_file_path = args.midi_file  # Get the path from the argument
wav_file_path = args.wav_file

input_file = args.wav_file
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

song_name = os.path.splitext(os.path.basename(input_file))[0]
output_dir = os.path.join('Spectrogram', song_name)

shape_freq_bins, complete_freq_bins = generate_spectrogram(input_file, output_dir)
print(shape_freq_bins)

#all_beats, beat_times = MIDI_analyzer.process_midi_file('Data/babyslakh_16k/Track00001/MIDI/S00.mid')
all_beats, beat_times, myTempo, myTime_signature = MIDI_analyzer.process_midi_file(midi_file_path)
for i in beat_times:
    #print(i)
    #print("This is our measure")
    for j in beat_times[i]:
        #print("This is our beat")
        #print(j)
        for my_note in beat_times[i][j]:
            #print(my_note.getNoteType())
            pass

data_in_input = []
for i in beat_times:
    spec_file_path = f"Spectrogram_Measures/measures_{i+1}"
    for j in beat_times[i]:
        spec_file_new = spec_file_path + f"spectrogram_{j}"
        note_data = []
        for my_note in beat_times[i][j]:
            cur_note_info = my_note.getData()
            #print(cur_note_info)

            note_data.append(my_note.getData())
        txtline = spec_file_new + ", " + str(note_data)
        data_in_input.append(txtline)

song_name = midi_file_path[5:len(midi_file_path)-4]

with open(song_name+".txt", "w") as file:
    for line in data_in_input:
        file.write(line+"\n")



# Go through the all_beats array by measure
# Mod the starting time by the time of the measure taking
# Group the starting based on what division based


output_midi = pretty_midi.PrettyMIDI(initial_tempo=myTempo)
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)
bps = myTempo/60
output_midi.time_signature_changes.append(pretty_midi.TimeSignature(myTime_signature[0], myTime_signature[1], time=0.0))


for i in range((len(all_beats))):
    start_time_array = {}
    # print(i)
    for my_note in all_beats[i].notes:
        start_time = my_note.getStartTime()
        if start_time in start_time_array:
            if start_time_array[start_time] == 1:
                start_time_array[start_time] += 1
                pitch = 64
            elif  start_time_array[start_time] == 2:
                start_time_array[start_time] += 1
                pitch = 67
            else:
                pitch = 60
        else:
            start_time_array[start_time] = 1
            pitch = 60

        duration = my_note.getDuration()
        #pitch = 60  # Middle C for simplicity
        velocity = 100  # Standard velocity

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time+(duration/bps))
        piano.notes.append(note)
        # print(my_note.getNoteType())

output_midi.instruments.append(piano)
output_dir = 'beat_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, f'{song_name}_output.mid')
output_midi.write(output_path)

def makeLSTMoutputDate(beat_times, spectrogram_timeframes, beat_per_measure, measure_duration, all_spect):
    num_freq_frames = (spectrogram_timeframes[(0, 0)][0])
    num_time_frames = (spectrogram_timeframes[(0, 0)][1])
    print("This is the measure_duration")
    print(measure_duration)
    print(beat_per_measure)
    frame_per_second = num_time_frames/measure_duration
    rhythm_array = np.zeros((len(beat_times), beat_per_measure, num_time_frames))
    with open("array_output", "w") as file:
        for row in beat_times:
            for col in beat_times[row]:
                for notes in beat_times[row][col]:
                    file.write(str(notes.getData()))
                file.write("\n")
            file.write("\n---------------\n")
        file.write("\n||||||||||||||||||\n")

    for measure_index in range(len(beat_times)):
        for beat_index in range(len(beat_times[measure_index])):
            #rhythm = np.zeros((spectrogram_timeframes[measure_index][beat_index])[0])
            notes = beat_times[measure_index][beat_index]
            for note in notes:
                note_in_measure_time = note.getStartTime() % measure_duration
                note_start_frame = int(note_in_measure_time * frame_per_second)
                note_end_frame = int((note_in_measure_time+note.getDuration()) * frame_per_second)
                if rhythm_array[measure_index, beat_index, note_start_frame] < 2:
                    rhythm_array[measure_index, beat_index, note_start_frame] = 2
                else:
                    rhythm_array[measure_index, beat_index, note_start_frame] +=1

                note_start_frame += 1

                remaining_time_frames = note_end_frame-note_start_frame
                loops_needed = ((note_start_frame - 1 + remaining_time_frames) // num_time_frames) + 1

                for extra_beat in range(loops_needed):
                    for frame in range(note_start_frame, min(remaining_time_frames, num_time_frames)):
                        rhythm_array[
                            measure_index+((beat_index+extra_beat)//beat_per_measure),
                            (beat_index+extra_beat) % beat_per_measure, frame
                        ] += 1
                    remaining_time_frames = remaining_time_frames - num_time_frames
                    note_start_frame = 0
    inputs = []
    outputs = []

    for measure_index in range(len(beat_times)):
        for beat_index in range(beat_per_measure):
            spectrogram_segment = all_spect[(measure_index, beat_index)]
            rhythm_segment = rhythm_array[measure_index][beat_index]
            inputs.append(spectrogram_segment)
            outputs.append(rhythm_segment)

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)

    return inputs_tensor, outputs_tensor


measure_duration = round(myTime_signature[0] / bps, 4)

input_tensor, output_tensor = makeLSTMoutputDate(beat_times, shape_freq_bins,
                                                 myTime_signature[0], measure_duration, complete_freq_bins)

#input_tensor = input_tensor.unsqueeze(1)  # This adds a channel dimension: [batch_size, 1, freq_bins, time_frames]

print(input_tensor.shape)
np.savetxt("see_what is in here", output_tensor, fmt='%d')
print(output_tensor.shape)

dataset = TensorDataset(input_tensor, output_tensor)

dataloader = DataLoader(dataset, batch_size=216)

spectrogram_shape = ((shape_freq_bins[(0, 0)])[0], (shape_freq_bins[(0, 0)])[1])
input_size = input_tensor.size(2)
hidden_size = 128
num_layers = 2
output_size = 15
batch_size = 216
learning_rate = 0.1
epochs = 30


model = Models.lstmModel.LSTM_model.SpectrogramRhythmModel(spectrogram_shape, hidden_size, num_layers, output_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0

    for batch_inputs, batch_outputs in dataloader:        # Get a batch of inputs and outputs
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_inputs)
        print(f"Shape of predictions: {predictions.shape}")
        print(f"Shape of predictions: {batch_outputs.shape}")


        # Compute the loss
        # Flatten the predictions and targets to match the shape for CrossEntropyLoss
        predictions = predictions.view(-1, output_size)  # Shape: [batch_size * time_frames, 3]
        batch_outputs = batch_outputs.view(-1)  # Shape: [batch_size * time_frames]
        batch_outputs = batch_outputs.long()
        print(f"Shape of predictions: {predictions.shape}")
        print(f"Shape of predictions: {batch_outputs.shape}")

        loss = loss_function(predictions, batch_outputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(input_tensor):.4f}")


torch.save(model.state_dict(), 'Models/lstmModel/trained_rhythm_model.pth')


