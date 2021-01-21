from glob import glob
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import concurrent.futures

from tensorflow.python.ops.math_ops import truediv

from MidiParser import MidiParser

def generateCSVFilesFromList(parser, file, composer):
    print("=== Working on Interpret: " + str(composer) + " and file: " + str(file))
    filepath = "src/datasets/midi_originals/" + composer + "/" + file
    song = parser.midiToArray(filepath)
    # ---- This will remove empty lines everywhere in the song, which will result in it having no silence in the end and beginning, as well as no full pause
    song_new = []
    for x in range(0,len(song)):
        sum = 0
        for y in range(1,89):
            sum += song[x][y]
        if sum != 0:
            song_new.append(song[x])
    # ----
    try:
        np.savetxt("src/datasets/arrays/" + composer + "/" + file + ".csv", song_new, fmt='%d', delimiter=';',
                           header='BPM;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C;;D;;E;F;;G;;A;;B;C')
    except:
        print("Filesystem Error writing CSV")
    
    return song

def generateAllCSVFiles():
    # get all folders in midi_originals
    folders = glob("src/datasets/midi_originals/*/")
    parser = MidiParser()
    # create folders in arrays
    for folder in folders:
        composer = folder.replace("src/datasets/midi_originals", "")
        composer = composer.strip("\\")
        fullfolder = "src/datasets/midi_originals/" + composer
        if not os.path.exists("src/datasets/arrays/" + composer):
            os.makedirs("src/datasets/arrays/" + composer)
        # iterate over folders in midi_originals and for each file
        # do the work and save the array in the new csv location
        files = [f for f in listdir(fullfolder) if isfile(join(fullfolder, f))]
        #generateCSVFilesFromList(parser, files, interpret)
        with concurrent.futures.ProcessPoolExecutor() as executor:
                
                futures = [executor.submit(
                    generateCSVFilesFromList, *[parser, file, composer]) for file  in files]

                results = [future.result() for future in futures]
                print(results)


if __name__ == "__main__":
    generateAllCSVFiles()
