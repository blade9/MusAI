import os
import ast
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []
        
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label_str = line.strip().split(',', 1)
                label = ast.literal_eval(label_str)  # Convert string to list
                self.data.append((img_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        
        # Load the image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return image, label_tensor


# Paths to your data
image_dir = 'path/to/spectrograms'
label_file = 'path/to/labels.txt'

# Create dataset and dataloader
dataset = SpectrogramDataset(image_dir=image_dir, label_file=label_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage in training loop
for images, labels in dataloader:
    print(images.shape)  # Batch of images
    print(labels)        # Corresponding labels

    hidden_size = 256
num_layers = 2
output_size = 88  # Adjust based on your label size
dropout = 0.2
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Create the model
model = SpectrogramRhythmModel(spectrogram_shape, hidden_size, num_layers, output_size, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create dataset and dataloader
train_dataset = SpectrogramDataset('path/to/train/spectrograms', 'path/to/train/labels.txt', transform=your_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for spectrograms, labels in train_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs.view(-1, output_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'spectrogram_rhythm_model.pth')

# Create test dataset and dataloader
test_dataset = SpectrogramDataset('path/to/test/spectrograms', 'path/to/test/labels.txt', transform=your_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        outputs = model(spectrograms)
        _, predicted = torch.max(outputs.data, 2)
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
