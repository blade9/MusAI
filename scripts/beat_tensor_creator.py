import numpy as np
from Models.lstmModel.NoteObject import NoteObject

def flatten_beat_objects(beat_objects):
    """
    Converts a list of BeatObject instances into training-ready tensors.

    Args:
    - beat_objects: List of BeatObject instances.

    Returns:
    - Tensor of shape (num_beats, time_frames, output_features).
    """
    flattened_data = []

    for beat in beat_objects:
        beat_data = []
        for note in beat.notes:
            note_vector = [
                note.getStartTime(),           # Starting time
                note.getDuration(),            # Duration
                NoteObject.NOTE_TYPES[note.getNoteType()]  # Encoded note type
            ]
            beat_data.append(note_vector)
        flattened_data.append(beat_data)

    # Pad sequences to ensure uniform shape for LSTM
    max_notes = max(len(beat.notes) for beat in beat_objects)
    flattened_data = [
        beat_data + [[0, 0, 0]] * (max_notes - len(beat_data)) for beat_data in flattened_data
    ]

    return np.array(flattened_data)  # Shape: (num_beats, max_notes, 3)
