import os
import ast
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from your_model_file import SpectrogramRhythmModel  # Import your model class

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

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to your data
image_dir = 'path/to/spectrograms'
label_file = 'path/to/labels.txt'

# Create dataset and dataloader
dataset = SpectrogramDataset(image_dir=image_dir, label_file=label_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hyperparameters
hidden_size = 256
num_layers = 2
output_size = 88  # Adjust based on your label size
dropout = 0.2
learning_rate = 0.001
num_epochs = 10
spectrogram_shape = (1, 1024, 55)  # Assuming this is your spectrogram shape

# Create the model
model = SpectrogramRhythmModel(spectrogram_shape, hidden_size, num_layers, output_size, dropout)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Use the existing train_model method
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train_model(dataloader, optimizer, criterion, num_epochs, device)

print("Training finished")
