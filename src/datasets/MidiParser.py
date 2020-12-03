from music21 import converter, stream, note, chord, tempo
import numpy as np
import math
from enum import IntEnum


class Sound(IntEnum):
    SILENCE = 0
    NOTESTART = 1
    NOTECONTINUED = 2


class MidiParser:

    arr = []
    length = 0

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

    def addKey(self, theNote, pos):
        # quarter length to 1/16 length and round
        duration = round(theNote.quarterLength * 4)
        pitchVal = MidiParser.LUT(theNote.pitch)
        self.arr[pos][pitchVal+1] = Sound.NOTESTART
        for i in range(1, duration):
            self.arr[pos+i][pitchVal+1] = Sound.NOTECONTINUED

    def midiToArray(self, filename):

        currentBPM = 120

        midif = converter.parse(filename)
        # Choosing Parts by Hand is done here
        rightPart = midif.parts[0]
        leftPart = midif.parts[1]
        theParts = [rightPart, leftPart]

        length = rightPart.highestTime
        if length < leftPart.highestTime:
            length = leftPart.highestTime
        self.length = math.ceil(length*4)  # quarters to 16ths

        self.arr = np.zeros((self.length, 88+1), dtype=np.int16)

        for i in range(0, self.length):
            for aPart in theParts:
                now = aPart.getElementsByOffset(
                    i/4, classList=['Note', 'Chord', 'MetronomeMark'])
                for element in now.recurse():

                    if isinstance(element, tempo.MetronomeMark):  # BPM Mark
                        currentBPM = element.number

                    if isinstance(element, note.Note):
                        self.addKey(element, i)

                    if isinstance(element, chord.Chord):
                        for n in element.notes:
                            self.addKey(n, i)
            self.arr[i][0] = currentBPM
        return self.arr