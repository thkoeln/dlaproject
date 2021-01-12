from glob import glob
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import concurrent.futures

from MidiParser import MidiParser

def generateCSVFilesFromList(parser, file, interpret):
    print("=== Working on Interpret: " + str(interpret) + " and file: " + str(file))
    filepath = "src/datasets/midi_originals/" + interpret + "/" + file
    song = parser.midiToArray(filepath)
    try:
        np.savetxt("src/datasets/arrays/" + interpret + "/" + file + ".csv", song, fmt='%d', delimiter=';',
                           header='BPM;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C')
    except:
        print("Filesystem Error writing CSV")

def generateAllCSVFiles():
    # get all folders in midi_originals
    folders = glob("src/datasets/midi_originals/*/")
    parser = MidiParser()
    # create folders in arrays
    for folder in folders:
        interpret = folder.replace("src/datasets/midi_originals", "")
        interpret = interpret.strip("\\")
        fullfolder = "src/datasets/midi_originals/" + interpret
        if not os.path.exists("src/datasets/arrays/" + interpret):
            os.makedirs("src/datasets/arrays/" + interpret)
        # iterate over folders in midi_originals and for each file
        # do the work and save the array in the new csv location
        files = [f for f in listdir(fullfolder) if isfile(join(fullfolder, f))]
        #generateCSVFilesFromList(parser, files, interpret)
        with concurrent.futures.ProcessPoolExecutor() as executor:
                
                futures = [executor.submit(
                    generateCSVFilesFromList, *[parser, file, interpret]) for file  in files]

                results = [future.result() for future in futures]
                print(results)


if __name__ == "__main__":
    generateAllCSVFiles()
