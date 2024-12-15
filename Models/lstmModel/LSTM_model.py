import torch
import torch.nn as nn
import torch.optim as optim

# Define spectrogram shape
# Format: (channels, frequency_bins, time_frames)
spectrogram_shape = (1, 1024, 55)


class SpectrogramRhythmModel(nn.Module):
    def __init__(self, spectrogram_shape, hidden_size, num_layers, output_size, dropout=0.2):
        super(SpectrogramRhythmModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d((2, 2))  # Reduces each dimension by half

        # Compute the flattened size after convolution + pooling
        channels, freq_bins, time_frames = spectrogram_shape
        self.lstm_input_size = (freq_bins // 2) * (time_frames // 2) * 64

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.lstm_input_size)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        output = self.output_layer(x)
        return output


# Hyperparameters
hidden_size = 256
num_layers = 2
output_size = 88  # Example: one-hot encoding for notes (88 piano keys)
dropout = 0.2

# Instantiate the model
model = SpectrogramRhythmModel(spectrogram_shape, hidden_size, num_layers, output_size)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example input
batch_size = 8
spectrogram = torch.randn(batch_size, *spectrogram_shape)  # Batch of spectrograms
labels = torch.randint(0, 88, (batch_size, spectrogram_shape[2]))  # Random labels for time frames

# Forward pass
output = model(spectrogram)
loss = criterion(output.view(-1, output_size), labels.view(-1))
print("Loss:", loss.item())
