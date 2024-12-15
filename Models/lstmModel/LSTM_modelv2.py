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
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Dense layer with ReLU
        dense_out = self.relu(self.dense(lstm_out))

        # Final output layer
        output = self.final_layer(dense_out)

        return output


# Hyperparameters
input_size = 128  # Input feature size (ensure your data matches this)
hidden_size = 256  # Size of the hidden layer in the LSTM
num_layers = 2  # Number of LSTM layers
output_size = 3  # The number of outputs (e.g., note duration, start time, type)
dropout = 0.2  # Dropout rate for regularization

# Instantiate the model
model = RhythmTranscriptionModel(input_size, hidden_size, num_layers, output_size, dropout)

# Optimizer and loss function
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Corrected to instantiate the loss function

# Dummy data for illustration (replace with your actual data)
X_train = torch.randn((32, 10, 128))  # 32 examples, 10 time steps, 128 features
y_train = torch.randn((32, 10, 3))  # 32 examples, 10 time steps, 3 output features


# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()

    # Forward pass
    optimizer.zero_grad()
    predictions = model(X_train)  # Forward pass through the model

    # Compute loss
    loss = criterion(predictions, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


