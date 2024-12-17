import os
import yaml
import pprint
import shutil

melodic = {'Guitar', 'Piano', 'Synth Lead'}
babyslakh_dir = '../data/babyslakh_16k/'
metadata_file_title = 'metadata.yaml'
extracted_stems_path = '../data/extracted_stems'


def get_metadata_stems(dir):
    global melodic
    track_stems = {}
    for entry in os.listdir(dir):
        entry_path = os.path.join(dir, entry)
        if os.path.isdir(entry_path):
            track = entry.title()
            track_stems[track] = []
            metadata_path = os.path.join(entry_path, metadata_file_title)

            with open(metadata_path, 'r') as metadata_file:
                metadata = yaml.safe_load(metadata_file)
            for stem, stem_metadata in metadata['stems'].items():
                if stem_metadata['inst_class'] in melodic:
                    track_stems[track].append(stem)
    return track_stems

# data_dir_path = '../babyslakh_16k/'
# for each track = data_dir_path + '{track}/stems/{wav}'
# new path = '../babyslakh_raw_stems_train/{wav}'
def extract_raw_data_for_training(data_dir_path, track_stems):
    i = 0
    for entry in os.listdir(data_dir_path):
        global extracted_stems_path
        entry_path = os.path.join(data_dir_path, entry)
        if os.path.isdir(entry_path):
            stems_path = os.path.join(entry_path, 'stems')
            midi_path = os.path.join(entry_path, 'MIDI')
            stems_to_extract = track_stems[entry.title()]
            for stem in stems_to_extract:
                stems_src = os.path.join(stems_path, f'{stem}.wav')
                stems_dest = os.path.join(extracted_stems_path, f'WAV/melody{i}.wav')
                midi_src = os.path.join(midi_path, f'{stem}.mid')
                midi_dest = os.path.join(extracted_stems_path, f'MIDI/melody{i}.mid')
                shutil.copy(stems_src, stems_dest)
                shutil.copy(midi_src, midi_dest)
                i += 1

if __name__ == '__main__':
    track_stems = get_metadata_stems(babyslakh_dir)
    extract_raw_data_for_training(babyslakh_dir, track_stems)