import os
from PIL import Image
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
sequence_length = 10
input_size = 128 * 128
hidden_size = 256
num_layers = 2
num_classes = 5
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Labels
def get_labels(num_sequences):
    # Load the audio file corresponding to the spectrograms
    audio_path = "Data/one-wish.mp3"  # Adjust path as needed
    y, sr = librosa.load(audio_path, sr=None)

    # Calculate onset envelope and tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, start_bpm=100, sr=sr)
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)

    # Convert tempo variations into discrete classes
    # We can use 5 classes based on tempo variations
    tempo_ranges = torch.linspace(dtempo.min(), dtempo.max(), num_classes + 1)

    # Create labels based on tempo variations
    labels = []
    for i in range(num_sequences):
        tempo_idx = i % len(dtempo)
        label = torch.tensor((dtempo[tempo_idx] > tempo_ranges[:-1]) &
                             (dtempo[tempo_idx] <= tempo_ranges[1:])).long()
        label = torch.argmax(label)
        labels.append(label)

    return torch.tensor(labels)

# Model definition
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


# Dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, sequence_length, transform=None):
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_files = sorted([
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith('.png')
        ])

        # Generate labels during initialization
        self.labels = get_labels(len(self.image_files) - sequence_length + 1)

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
        return sequence, self.labels[idx]


# Data transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = SpectrogramDataset('Spectrogram/one-wish', sequence_length, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = RhythmLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

labels = get_labels(len(dataset))

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i, (sequences, batch_labels) in enumerate(dataloader):
        sequences = sequences.view(sequences.size(0), sequence_length, -1).to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'rhythm_lstm_model.pth')