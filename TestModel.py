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
import Models.lstmModel.LSTM_model

import os, argparse

parser = argparse.ArgumentParser(description="Process a WAV file and generate a MIDI output.")
parser.add_argument("wav_file", type=str, help="Path to the original WAV file of the song")
args = parser.parse_args()
input_file = args.wav_file
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

song_name = os.path.splitext(os.path.basename(input_file))[0]
output_dir = os.path.join('New_Spectrogram', song_name)

shape_freq_bins, complete_freq_bins = generate_spectrogram(input_file, output_dir)
inputs = []

for key in complete_freq_bins:
    spectrogram_segment = complete_freq_bins[key]
    inputs.append(spectrogram_segment)

inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

spectrogram_shape = ((shape_freq_bins[(0, 0)])[0], (shape_freq_bins[(0, 0)])[1])
input_size = inputs_tensor.size(2)
hidden_size = 128
num_layers = 2
output_size = 15
batch_size = 216
learning_rate = 0.001
epochs = 300
model = Models.lstmModel.LSTM_model.SpectrogramRhythmModel(spectrogram_shape, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('Models/lstmModel/trained_rhythm_model.pth'))
model.eval()
with torch.no_grad():  # No need to compute gradients for inference
    predictions = model(inputs_tensor)

# If needed, you can get the predicted class labels (for classification tasks)
predicted_classes = torch.argmax(predictions, dim=-1)  # Choose the class with the highest probability
print(predicted_classes)
np.savetxt("new_file_generated", predicted_classes, fmt='%d')