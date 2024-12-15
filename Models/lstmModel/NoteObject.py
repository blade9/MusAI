
class NoteObject:

    def __init__(self, duration, start_time, time_num=4, time_denom=4):
        self.time_signature = time_num, time_denom
        self.duration = duration
        self.note_type = self.calcNoteType(duration/time_denom)
        self.starting_time = start_time%time_num


    NOTE_TYPES = {
        "whole": 1,
        "half": 0.5,
        "quarter": 0.25,
        "eighth": 0.125,
        "sixteenth": 0.06125,
        "thirty-second": 0.03075
    }

    def getNoteType(self):
        return self.note_type

    def calcNoteType(self, duration_in_beats):
        for note_type, threshold in NoteObject.NOTE_TYPES.items():
            if duration_in_beats >= threshold:
                return note_type
        return "none"




