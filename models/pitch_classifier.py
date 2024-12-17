import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN architecture
def build_pitch_classifier_cnn(input_shape=(1025, 173, 1), num_pitches=128):
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(16, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Output: (512, 86, 16)

    # Convolutional Layer 2
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Output: (256, 43, 32)

    # Convolutional Layer 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))  # Output: (128, 21, 64)

    # Flattening Layer
    model.add(layers.Flatten())  # Output: flattened features

    # Fully Connected Layers
    model.add(layers.Dense(512, activation='relu'))  # FC layer 1
    model.add(layers.Dense(num_pitches, activation='sigmoid'))  # FC layer 2 (Output)

    return model
