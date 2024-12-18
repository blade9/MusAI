import tensorflow as tf
import numpy as np
import os
import ast
from sklearn.model_selection import train_test_split
from Models.lstmModel.LSTM_modelv2 import create_spectrogram_rhythm_model

class SpectrogramDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_file, batch_size=32, image_size=(224, 224)):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.data = []
        
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label_str = line.strip().split(',', 1)
                label = ast.literal_eval(label_str)  # Convert string to list
                self.data.append((img_name, label))
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for img_name, label in batch_data:
            img_path = os.path.join(self.image_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.image_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
            batch_images.append(img_array)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        np.random.shuffle(self.data)

# Create the full dataset
full_dataset = SpectrogramDataset('path/to/spectrograms', 'path/to/labels.txt', batch_size=32)

# Split the data into train and validation sets
train_data, val_data = train_test_split(full_dataset.data, test_size=0.2, random_state=42)

hidden_size = 256
num_layers = 2
output_size = 88  # Example: one-hot encoding for notes (88 piano keys)
dropout = 0.2

# Create and compile the model
model = create_spectrogram_rhythm_model((224, 224, 3), hidden_size, num_layers, output_size, dropout)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

print("Training finished")

# Save the model
model.save('spectrogram_rhythm_model.h5')

