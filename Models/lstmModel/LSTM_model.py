import torch
import torch.nn as nn

import torch.optim as optim
from torchvision import models

# Define spectrogram shape
# Format: (channels, frequency_bins, time_frames)

class SpectrogramRhythmModel(nn.Module):
    def __init__(self, spectrogram_shape, hidden_size, num_layers, output_size, dropout=0.2):
        super(SpectrogramRhythmModel, self).__init__()
        '''
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Unpack the spectrogram shape
        freq_bins, time_frames = spectrogram_shape

        # Reshape the input for LSTM (we want [batch_size, time_frames, freq_bins])
        self.lstm = nn.LSTM(input_size=freq_bins, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        '''
        freq_bins, time_frames = spectrogram_shape

        self.lstm = nn.LSTM(input_size=freq_bins, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        '''
        batch_size, _, time_frames = x.size()

        # Ensure correct shape [batch_size, time_frames, freq_bins]
        x = x.view(batch_size, time_frames, -1, out)  # Flatten the frequency dimension

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output shape: [batch_size, time_frames, hidden_size]

        # Pass through fully connected layers
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step for classification

        x = self.fc1(lstm_out)  # First fully connected layer
        x = torch.relu(x)  # ReLU activation
        output = self.fc2(x)  # Output layer
        return output
        '''
        batch_size, freq_bins, time_frames = x.size()
        x = x.view(batch_size, time_frames, freq_bins)

        lstm_out, __ = self.lstm(x)
        x = self.fc1(lstm_out)  # First fully connected layer
        x = torch.relu(x)  # ReLU activation
        output = self.fc2(x)  # Output layer
        return output



''' 

        # CNN layers
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        channels, freq_bins, time_frames = spectrogram_shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, freq_bins, time_frames)  # 1 batch, 1 channel
            conv_out = self.pool(self.relu(
                self.conv2(self.pool(self.relu(self.conv1(dummy_input))))))  # [1, 64, freq_bins/4, time_frames/4]
            _, _, flattened_freq_bins, flattened_time_frames = conv_out.shape
            self.flattened_size = flattened_freq_bins * 64  # Multiply the number of channels (64) by the frequency bins after pooling

        self.lstm = nn.LSTM(
            input_size=self.flattened_size,  # Input size to the LSTM comes from the CNN output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
        # LSTM layers
        self.lstm = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        #nn.LSTM(input_size=self.flattened_size, hidden_size=hidden_size,
         #                   num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu()
        #self.output_layer = nn.Linear(hidden_size // 2, output_size)
'''

    #def forward(self, x):
'''
        # Apply convolutional layers
        # Apply convolutional layers
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 32, freq_bins, time_frames]
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 64, freq_bins/4, time_frames/4]

        # Flatten the CNN output for LSTM
        batch_size, channels, freq_bins, time_frames = x.size()
        x = x.permute(0, 3, 1, 2)  # Rearrange to [batch_size, time_frames, channels, freq_bins]
        x = x.contiguous().view(batch_size, time_frames, -1)  # [batch_size, time_frames, input_size]

        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len (time_frames), hidden_size]

        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))  # [batch_size, seq_len, hidden_size // 2]
        output = self.fc2(x)  # [batch_size, seq_len, output_size]

        return output


        x = self.pool(self.relu(self.conv2(self.pool(self.relu(self.conv1(x))))))  # Apply conv1, conv2, and pooling
        batch_size, channels, freq_bins, time_frames = x.size()
        x = x.permute(0, 3, 1, 2)  # Rearrange to [batch_size, time_frames, channels, freq_bins]
        x = x.reshape(batch_size, time_frames, -1)  # Flatten channels and freq_bins for LSTM input

        # Flatten the output to match the input size for LSTM

        #flattened_size = channels * freq_bins
        #x = x.view(batch_size, time_frames, flattened_size)  # Reshaping for LSTM: [batch_size, seq_len, input_size]

        # LSTM processing
        if self.lstm is None:
            # Initialize LSTM only after knowing the flattened size
            self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True, dropout=0.2)

        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)


        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        output = self.output_layer(x)
        return output 
        
'''


