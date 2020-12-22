class TestNote:
    def __init__(self, tone, start, length):
        self.tone = tone
        self.length = length
        self.start = start


class TestChord:
    def __init__(self, start, length):
        self.notes = []
        self.length = length
        self.start = start

    def add(self, note):
        if note.length == self.length and note.start == self.start:
            self.notes.append(note.tone)
            return True
        else:
            return False


class TestTrack:
    def __init__(self):
        self.chords = []

    def active(self, time):
        for i in range(0, len(self.chords)):
            if self.chords[i].start + self.chords[i].length >= time:
                return True
        return False

    def add(self, chord: TestChord):
        if self.checkactive(chord.start):
            return False
        else:
            self.chords.append(chord)
            return True


def split(arr):
    tracks: []
    for i in range(0, len(arr)):
        notes = []
        for j in range(1, 89):
            if arr[i][j] == 1:
                k = i + 1
                while arr[k][j] == 2:
                    k += 1
                notes.append(TestNote(j, i,k-i))
        chords = []
        for n in range(0, len(notes)):
            added = False
            for a in range(0, len(chords)):
                if chords[a].add(notes[n]):
                    added = True
            if not added:
                chord = TestChord(notes[n].start, notes[n].start)
                chord.add(notes[n])
                chords.append(chord)

        for y in range(0, len(chords)):
            added = False
            for z in range(0, len(tracks)):
                if tracks[z].add(chords[y]):
                    added = True
            if not added:
                track = TestTrack()
                track.add(chords[y])
                tracks.append(track)
