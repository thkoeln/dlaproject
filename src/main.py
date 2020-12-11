from datasets import MidiParser
import os


def main():
    parser = MidiParser.MidiParser()
    arr = parser.midiToArray(os.getcwd() + "/datasets/midi_originals/balakirew/islamei_format0.mid")
    for row in arr:
        print(row)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
