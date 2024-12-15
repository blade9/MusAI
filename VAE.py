import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder: Extract features from input
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),  # Convolutional layer
            nn.ReLU(),
            nn.MaxPool1d(2),  # Downsample
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),  # Convolutional layer
            nn.ReLU(),
            nn.MaxPool1d(2),  # Downsample
        )

        self.lstm = nn.LSTM(256, latent_dim, num_layers=2, batch_first=True)
        self.fc_latent = nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Ensure the output matches the input scale (e.g., normalized spectrogram)
        )

        input_dim = 64  # Number of frequency bins in the spectrogram
        latent_dim = 128  # Size of the latent space
        learning_rate = 0.001
        num_epochs = 50

        model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        train_loader = torch.utils.data.DataLoader( )

        def forward(self, x):
            # Encoder
            x = self.encoder(x)
            x = x.permute(0, 2, 1)  # Prepare for LSTM (batch, time, features)

            # LSTM
            x, _ = self.lstm(x)
            latent = self.fc_latent(x[:, -1, :])  # Take the final hidden state

            # Decoder (Reconstruction)
            recon = self.decoder(latent)
            return recon, latent

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                recon, latent = model(batch)

                # Compute reconstruction loss
                loss = loss_function(recon, batch)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)  # Latent mean
        self.fc_logvar = nn.Linear(128, latent_dim)  # Latent log variance

        # Decoder
        self.fc2 = nn.Linear(latent_dim, 128)
        self.fc3 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Sample epsilon
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # Output in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar