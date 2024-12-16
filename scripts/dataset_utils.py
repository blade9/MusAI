from dataset_generator import *
import tensorflow as tf

MIDI_FILE_PATH = '../data/extracted_stems/MIDI'
TFRECORD_PATH = '../data/processed_data/dataset.tfrecord'

def serialize_example(spectrogram, label):
    feature = {
        'spectrogram': tf.train.Feature(float_list=tf.train.FloatList(value=spectrogram.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def save_as_tfrecord(dataset, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for spectrogram, label in dataset.unbatch():
            serialized_example = serialize_example(spectrogram.numpy(), label.numpy())
            writer.write(serialized_example)

def parse_tfrecord(example_proto):
    # Define the feature structure
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([1025 * 173], tf.float32),  # Adjust to your data shape
        'label': tf.io.FixedLenFeature([128], tf.int64),
    }

    # Parse the serialized example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Reshape the spectrogram to its original shape
    spectrogram = tf.reshape(parsed_features['spectrogram'], [1025, 173])
    label = parsed_features['label']  # Labels don't need reshaping

    return spectrogram, label

def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    # Create a raw TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse the TFRecord examples
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    # Shuffle, batch, and prefetch for training
    dataset = (
        parsed_dataset
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


if __name__ == '__main__':
    dataset = load_tfrecord_dataset(TFRECORD_PATH)
    for spectrogram, label in dataset.take(10):
        print(f"Spectrogram shape: {spectrogram.shape}")
        print(f"Label shape: {label.shape}")

