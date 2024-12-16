import os

from scripts.preprocessingutils import *
import tensorflow as tf


def datapoint_generator(midi_file_path):
    grouped_notes = group_pitches_by_onset(midi_file_path)
    unique_groups = remove_redundant_groups(grouped_notes)
    for start_time, notes in unique_groups.items():
        midi = create_midi_from_group(start_time, notes)
        spectrogram = midi_to_spectrogram(midi)
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        label = get_midi_label(midi)
        yield spectrogram.astype(np.float32), label.astype(np.int32)


def create_tf_dataset(midi_file_path):
    def multi_file_generator(midi_file_path):
        midi_files = [f for f in os.listdir(midi_file_path)]

        for midi in os.listdir(midi_file_path):
            midi_file = os.path.join(midi_file_path, midi)
            print(f'Processing {midi.decode('utf-8')}...')
            full_path = os.path.join(midi_file_path, midi)
            yield from datapoint_generator(full_path)

    output_signature = (
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(128,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        multi_file_generator,
        args=[midi_file_path],
        output_signature=output_signature,
    )

    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset