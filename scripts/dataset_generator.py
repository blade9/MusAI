import tensorflow as tf
import numpy as np
import os

def load_data_from_file(file_path):
    filenames = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            filename, *label_values = line.strip().split()
            filenames.append(filename)
            labels.append([float(val) for val in label_values])
    return filenames, np.array(labels)

def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def generate_dataset(input_file='input.txt', batch_size=32, augment=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Get the directory of the input file
    input_dir = os.path.dirname(input_file)

    # Load filenames and labels
    filenames, labels = load_data_from_file(input_file)

    # Prepend the input directory to the filenames
    filenames = [os.path.join(input_dir, filename) for filename in filenames]

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# This will be called when the module is reloaded
dataset = generate_dataset()
