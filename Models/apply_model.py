import os
import torch
import torch.nn as nn
from PIL import Image
import librosa
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
sequence_length = 10
input_size = 128 * 128
hidden_size = 256
num_layers = 2
num_classes = 5  # 12 notes in an octave
learning_rate = 0.001


class RhythmLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RhythmLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def get_note_from_freq(frequency):
    """Convert frequency to musical note"""
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    if frequency == 0:
        return "Silence"

    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    n = h % 12
    return note_names[n]


class NewSongDataset(Dataset):
    def __init__(self, image_dir, sequence_length, transform=None):
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_files = sorted([
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith('.png')
        ], key=lambda x: int(os.path.basename(x).split('spectrogram')[1].split('.')[0]))

    def __len__(self):
        return len(self.image_files) - self.sequence_length + 1

    def __getitem__(self, idx):
        images = []
        for i in range(idx, idx + self.sequence_length):
            image_path = self.image_files[i]
            image = Image.open(image_path).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        sequence = torch.stack(images)
        sequence = sequence.view(self.sequence_length, -1)
        return sequence


def analyze_song_notes(song_name):
    # Load the trained model
    model = RhythmLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load('rhythm_lstm_model.pth'))
    model.eval()

    # Load audio file for frequency analysis
    audio_path = f"Data/{song_name}.mp3"
    y, sr = librosa.load(audio_path)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = NewSongDataset(f'Spectrogram/{song_name}', sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Analysis results
    rhythm_notes = []

    with torch.no_grad():
        for i, sequences in enumerate(dataloader):
            sequences = sequences.view(sequences.size(0), sequence_length, -1).to(device)
            outputs = model(sequences)

            # Get the predominant frequency for this segment
            start_sample = int(i * 0.5 * sr)
            end_sample = int((i + 1) * 0.5 * sr)
            if end_sample <= len(y):
                segment = y[start_sample:end_sample]
                if len(segment) > 0:
                    frequencies = librosa.piptrack(y=segment, sr=sr)[1]
                    if len(frequencies.flatten()) > 0:
                        main_freq = np.mean(frequencies[frequencies > 0]) if np.any(frequencies > 0) else 0
                        note = get_note_from_freq(main_freq)
                        rhythm_notes.append(note)

    # Print analysis
    print(f"\nNote Analysis for {song_name}:")
    print("--------------------------------")

    for i, note in enumerate(rhythm_notes):
        time_point = i * 0.5
        print(f"Time {time_point:.1f}s - {time_point + 0.5:.1f}s: {note}")

    # Calculate note distribution
    unique_notes = set(rhythm_notes)
    note_distribution = {}
    total_notes = len(rhythm_notes)

    for note in unique_notes:
        count = rhythm_notes.count(note)
        percentage = (count / total_notes) * 100
        note_distribution[note] = percentage

    print("\nNote Distribution:")
    print("------------------")
    for note, percentage in sorted(note_distribution.items()):
        print(f"{note}: {percentage:.1f}%")

    # Identify dominant note progression
    print("\nDominant Note Progression:")
    print("-------------------------")
    window_size = 4
    for i in range(0, len(rhythm_notes) - window_size + 1, window_size):
        progression = ' -> '.join(rhythm_notes[i:i + window_size])
        time_start = i * 0.5
        time_end = (i + window_size) * 0.5
        print(f"{time_start:.1f}s - {time_end:.1f}s: {progression}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze note patterns in a song')
    parser.add_argument('song_name', type=str, help='Name of the song to analyze')
    args = parser.parse_args()

    analyze_song_notes(args.song_name)