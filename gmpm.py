import h5py
import numpy as np
import os


def load_h5_dataset(directory):
    print(" --------------------------------- ")
    print("Start loading Datasat from H5DF files...")
    data = []
    flagOneFile = 0
    for filename in os.listdir(directory):
        if flagOneFile:
            break
        if filename.endswith(".h5"):
            with h5py.File(filename, "r") as f:
                a_group_key = list(f.keys())[0]
                # Get the data
                temp = list(f[a_group_key])
                data.append(temp[1:])
                flagOneFile = 0
            continue
        else:
            continue
    data_flat = [item for sublist in data for item in sublist]
    data_flat = np.stack(data_flat, axis=0)
    precent_train_test_split = 0.7
    train = data_flat[:int(np.floor(precent_train_test_split * data_flat.shape[0])), :]
    test = data_flat[int(np.floor(precent_train_test_split * data_flat.shape[0])) + 1:, :]

    if not os.path.isfile('test_imagegpt.h5'):
        print("Saving H5DF files...")
        test_h5 = h5py.File('test_imagegpt.h5', 'w')
        test_h5.create_dataset('test', data=test)
        train_h5 = h5py.File('train_imagegpt.h5', 'w')
        train_h5.create_dataset('train', data=train)

    print(" --------------------------------- ")
    print("Finish loading Datasat from H5DF files...")

    return train, test

directory = "./"

train, test = load_h5_dataset(directory)