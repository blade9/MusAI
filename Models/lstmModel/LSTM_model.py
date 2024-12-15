import torch
import torch.nn as nn
import torch.optim as optim


class RhythmTranscriptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(RhythmTranscriptionModel, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer to map LSTM outputs to a higher-dimensional space
        self.dense = nn.Linear(hidden_size, hidden_size // 2)

        # Non-linearity (ReLU)
        self.relu = nn.ReLU()

        # Final output layer to predict rhythmic features (e.g., note duration, start times)
        self.final_layer = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        dense_out = self.relu(self.dense(lstm_out))
        output = self.final_layer(dense_out)
        return output

input_size = 128
hidden_size = 256
num_layers = 2
output_size = 3
dropout = 0.2

model = RhythmTranscriptionModel(input_size, hidden_size, num_layers, output_size, dropout)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss