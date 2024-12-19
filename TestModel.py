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

parser = argparse.ArgumentParser(description="Process a MIDI file and generate a MIDI output.")
parser.add_argument("wav_file", type=str, help="Path to the original WAV file of the song")
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


