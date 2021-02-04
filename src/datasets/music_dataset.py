import tensorflow as tf
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

basepath = "src/datasets/arrays/"

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BASE_BPM = 100.0
BPM_MODIFIER = 100.0
# input/output size (for us=(88)*3 + 1  = 265)
FEATURE_SIZE = 177

#  Is actually seems to do data windowing (@see https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing)
# bisher verarbeitete samples


def data_windowing(dataset, target, start_index: int, end_index: int, history_size: int,
                   target_size: int, step: int, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def get_dataset(batch_size=32, buffer_size=10000, train_split_pct=0.5, seed=13, debug=True, plot=False, past_history=1024, future_target=64, step_size=16, single_step=True, composer=None):
    # Load Dataset from csv to arrays (filtered by composer)
    dataset_csv_files = []
    if composer != None:
        _, _, csv_files = next(os.walk(basepath + composer))
        csv_filenames_after = []
        for csv_file in csv_files:
            csv_file = basepath + composer + "/" + csv_file
            csv_filenames_after.append(csv_file)
        dataset_csv_files.extend(csv_filenames_after)
    else:
        _, composers, _ = next(os.walk(basepath))
        for composer in composers:
            for (dirpath, dirnames, filenames) in os.walk(basepath + composer):
                csv_filenames_after = []
                for csv_file in filenames:
                    csv_file = basepath + composer + "/" + csv_file
                    csv_filenames_after.append(csv_file)
                dataset_csv_files.extend(csv_filenames_after)
                break

    if debug:
        print(dataset_csv_files[:10])

    complete_dataframe_set = pd.DataFrame()
    for dataset_csv_file in dataset_csv_files:
        dataframe = pd.read_csv(dataset_csv_file, delimiter=";")
        if debug:
            print("Creating Pandas DataFrame for: " + dataset_csv_file)
            print("First Line: " + str(dataframe.to_numpy()[0])) # TODO: This sometimes contains NaN??? How come?
        complete_dataframe_set = complete_dataframe_set.append(dataframe)

    if debug:
        print(complete_dataframe_set.head(10))
        # Also get first line completely:
        print(complete_dataframe_set.to_numpy()[0])
        print(complete_dataframe_set.to_numpy()[16])
        print(complete_dataframe_set.to_numpy()[32])
        print(complete_dataframe_set.to_numpy()[64])
        print(complete_dataframe_set.to_numpy()[128])

    # set random seed
    tf.random.set_seed(seed)

    # get the data from the dataset and define the features (metronome and notes) + normalization to float values
    features = complete_dataframe_set.to_numpy()
    features_extended = np.zeros(
        (features.shape[0], FEATURE_SIZE), dtype=np.float)
    if debug:
        print("Features Shape: {}".format(features.shape))
        print("Amount of 16th-Note-Rows in Dataset: " + str(features.shape[0]))
        print("Iterate over these...")
    for x in range(features.shape[0]):
        features_extended[x][0] = (features[x][0]-BPM_MODIFIER)/BASE_BPM
        if x%10000 == 0:
            print("{} rows transformed...".format(x))
        for y in range(1, 89):
            if features[x][y] == 0:
                #    features_extended[x][y*3 - 2] = 1.0 # Reducing this value does not help training
                continue
            if features[x][y] == 1:
                features_extended[x][y*2 - 1] = 1.0
                continue
            if features[x][y] == 2:
                features_extended[x][y*2+1 - 1] = 1.0
                continue
            print(
                "*** ERROR on feature normalization: There are values not fitting here ***")

    features = None
    if debug:
        print(features_extended[0])
        print(features_extended[16])
        print(features_extended[32])
        print(features_extended[64])
        print(features_extended[128])

    # normalize data (splitting per amount of notes etc)

    # split for train and validation set
    #dataset = features.values
    dataset = features_extended
    dataset_size = dataset.shape[0]
    print("Dataset contains {} rows, splitting by {}%".format(
        dataset_size, train_split_pct*100.0))
    train_split = int(train_split_pct*dataset_size)
    # ??? vvv was macht das?
    #data_mean = dataset[:train_split].mean(axis=0)
    #data_std = dataset[:train_split].std(axis=0)

    #dataset = (dataset - data_mean) / data_std

    train_set = dataset[:train_split]
    test_set = dataset[train_split:]
    

    # ??? ^^^

    train_data_gen = TimeseriesGenerator(train_set, train_set,
                                         length=past_history, sampling_rate=1, stride=1,
                                         batch_size=batch_size)


    test_data_gen = TimeseriesGenerator(test_set, test_set,
                                    length=past_history, sampling_rate=1, stride=1,
                                    batch_size=batch_size)

    # TODO: Check this, is this feasible here?
    # x_train_single, y_train_single = data_windowing(dataset, dataset, 0,
    #                                                   train_split, past_history,
    #                                                   future_target, step_size,    
    #                                                   single_step=single_step)
    # x_val_single, y_val_single = data_windowing(dataset, dataset,
    #                                               train_split, None, past_history,
    #                                               future_target, step_size,
    #                                               single_step=single_step)

    # debug output
    # if debug:
    #    print('Single window of past history : {}'.format(x_train_single[0].shape))

    # if plot:
    #    features.plot(subplots=True)

    # transform to tensorflow dataset
    #train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    #train_data_single = train_data_single.cache().shuffle(buffer_size).batch(batch_size)  # .repeat()

    #val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    #val_data_single = val_data_single.batch(buffer_size)  # .repeat()

    #return train_data_single, val_data_single, x_train_single.shape[-2:]
    return train_data_gen, test_data_gen, train_set.shape[-2:], test_set


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    get_dataset()
