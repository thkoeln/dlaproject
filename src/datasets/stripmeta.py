# This file is responible to cut information that is not needed from the MIDI files and save them in a different folder

from os import listdir, rename
from os.path import isfile, join, abspath, splitext
import mido
from mido import Message, MidiFile, MidiTrack, tempo2bpm, tick2second

def sanitizeFilenames(fileList) -> list:
    """
    Removes special characters and empty spaces from filenames for easier handling
    """
    sanitizedFiles = []
    for file in fileList:
        file = str(file)
        newname = file.lower()
        newname = newname.replace(" ","")
        newname = newname.replace("-","_")
        if (file != newname):
            rename(file, newname)
        sanitizedFiles.append(newname)

    return sanitizedFiles

def getOriginalMidis() -> list:
    """
    Loads up all original midi files in a list and returns this list
    """
    midifiles = [f for f in listdir("./midi_originals") if isfile(join("./midi_originals", f)) and
                 splitext(f)[1] == ".mid"]
    print(midifiles)
    midi_fullpath = []
    for file in midifiles:
        midi_fullpath.append(abspath(join("./midi_originals", file)))

    return midi_fullpath

def stripMidiMeta():
    """
    Reduces the Midi Files to the needed information
    """
    midifiles = getOriginalMidis()
    midifiles = sanitizeFilenames(midifiles)
    print(midifiles)
    for file in midifiles:
        midifile = mido.MidiFile(file)
        print("Midi File:\t\t" + str(file))
        print("Midi File Type:\t\t" + str(midifile.type))
        print("Midi Length (s):\t" + str(midifile.length))
        # Beats Per Minute are irrelevant, as we can change that later on the output without changing the way the music works

        # TODO: Create new midi file with the right (amount of) channels

        for msg in midifile:
            print(msg)
            if msg.type == "note_off" or msg.type == "note_on":
                # print(msg)
                # TODO: Add to new midi file
                pass
            

if __name__ == '__main__':
    stripMidiMeta()