import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define spectrogram shape
# Format: (channels, frequency_bins, time_frames)
spectrogram_shape = (1, 1024, 55)


class SpectrogramRhythmModel(nn.Module):
    def __init__(self, spectrogram_shape, hidden_size, num_layers, output_size, dropout=0.2):
        super(SpectrogramRhythmModel, self).__init__()

        # CNN layers

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        # Compute the flattened size after convolution + pooling
        channels, freq_bins, time_frames = spectrogram_shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, freq_bins, time_frames)  # 1 batch, 1 channel
            conv_out = self.pool(self.relu(self.conv2(self.pool(self.relu(self.conv1(dummy_input))))))
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.lstm_input_size = (freq_bins // 2) * (time_frames // 2) * 64

        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)


        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # ResNet feature extraction
        x = self.resnet(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the ResNet output

        # LSTM processing
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension

        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        output = self.output_layer(x)
        return output
    
    def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
        model.to(device)
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            for spectrograms, labels in dataloader:
                # Move data to the appropriate device (CPU/GPU)
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(spectrograms)

                # Reshape outputs and labels for loss computation
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print epoch loss
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


