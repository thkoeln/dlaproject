from music21 import converter, instrument, stream, note, chord, midi, tempo, meter
import numpy as np


def LUT(pitch):
    '''Turns a music21.pitch object into an Integer between 0 and 87 according to the corresponding piano key'''
    out = 0
    out += (pitch.octave * 12)

    offsetKey = {
        'C': -9,
        'D': -7,
        'E': -5,
        'F': -4,
        'G': -2,
        'A': 0,
        'B': 2
    }.get(pitch.step)

    out += offsetKey
    if pitch.accidental is not None:
        if pitch.accidental.name == 'sharp':
            out += 1
        if pitch.accidental.name == 'flat':
            out += -1
    return out


def rLUT(key):
    '''Turns an Integer between 0 and 87 into a String of the corresponding key name'''
    octave = 0
    offset = 0
    if key == 0:
        return 'A0'
    if key == 1:
        return 'A#0'
    if key == 2:
        return 'B0'
    if key >= 3:
        octave = 1
        offset = 3
    if key >= 15:
        octave = 2
        offset = 15
    if key >= 27:
        octave = 3
        offset = 27
    if key >= 39:
        octave = 4
        offset = 39
    if key >= 51:
        octave = 5
        offset = 51
    if key >= 63:
        octave = 6
        offset = 63
    if key >= 75:
        octave = 7
        offset = 75
    if key == 87:
        octave = 8
        offset = 87

    key = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'D#',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'G#',
        9: 'A',
        10: 'A#',
        11: 'B'
    }.get(key-offset)

    return key+str(octave)


def parseKey(key, duration, bpm):
    #out = np.zeros((duration, 88))
    out = np.zeros(88+2)
    out[0] = duration
    out[1] = bpm
    if key != -1:
        #out[0, key] = 1
        out[key+2] = 1
    # for i in range(1, duration-1):    #Mit Codierung 0.5
        #out[i, key] = 0.5
    return out


def parseChord(keys, duration, bpm):
    #out = np.zeros((duration, 88))
    out = np.zeros(88+2)
    out[0] = duration
    out[1] = bpm
    for key in keys:
        #out[0, key] = 1
        out[key+2] = 1
    # for i in range(1, duration-1):    #Mit Codierung 0.5
        #out[i, key] = 0.5
    return out


def midiToArray(filename):

    #arr = np.zeros(88, dtype=np.int8)
    arr = np.zeros(88+2, dtype=np.int8)
    currentBPM = 120
    durs = []  # Debugging

    midif = converter.parse(filename)

    for element in midif.parts[0]:#midif.flat:#midif.notesAndRests:

        # quarter length to 1/16 length and round
        duration = round(element.quarterLength * 4)

        if 1 == 1:  # duration > 0:

            if isinstance(element, tempo.MetronomeMark):  # BPM Mark
                currentBPM = element.number

            if isinstance(element, note.Note):  # Note
                arr = np.vstack(
                    (arr, parseKey(LUT(element.pitch), duration, currentBPM)))
                durs.append(element.duration.quarterLength)
                # print(element.name + ' ' + str(element.quarterLength))  #verbose

            if isinstance(element, note.Rest):  # Pause
                arr = np.vstack(
                    (arr, parseKey(-1, duration, currentBPM)))
                durs.append(element.duration.quarterLength)
                # print(element.name + ' ' + str(element.quarterLength))  #verbose

            if isinstance(element, chord.Chord):  # Chord
                pitches = []
                for theNote in element.notes:
                    pitches.append(LUT(theNote.pitch))
                arr = np.vstack(
                    (arr, parseChord(pitches, duration, currentBPM)))
                durs.append(element.duration.quarterLength)
                # print(element.fullName) #verbose

    print(np.unique(durs))  # Debugging
    return arr


def arraytoMidi(arr, filename):

    prevBPM = 0
    s = stream.Stream()
    s.append(meter.TimeSignature('4/4'))
    for theNote in arr:
        if theNote[1] != prevBPM:
            s.append(tempo.MetronomeMark(number=theNote[1]))
            prevBPM = theNote[1]
        keys = []
        n = 0
        for i in range(0, 88):
            if theNote[i+2] == 1:
                keys.append(rLUT(i))
        if len(keys) == 1:
            n = note.Note(keys[0])
        if len(keys) == 0:
            n = note.Rest()
        if len(keys) > 1:
            n = chord.Chord(keys)
        n.duration.quarterLength = theNote[0]/4
        s.append(n)
    s.write('midi', fp=filename)


def main():

    # mf = midi.MidiFile()
    # mf.open('src/datasets/midi_originals/Super Mario 64 - Medley.mid')
    # mf.read()
    # mf.close()
    # theStream = midi.translate.midiFileToStream(mf)
    song = midiToArray('src/datasets/midi_originals/Pirates of the Caribbean.mid')
    arraytoMidi(song,'src/datasets/midi_originals/Pirates of the Caribbean-parsed.mid')


if __name__ == "__main__":
    main()
