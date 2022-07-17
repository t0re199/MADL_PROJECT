import os
import pickle

from Constants import CHECKPOINTS_PATH


def save_object(data, file, directory=CHECKPOINTS_PATH):
    with open(os.path.join(directory, file), 'wb') as file:
        pickle.dump(data, file)


def load_object(file, directory=CHECKPOINTS_PATH):
    with open(os.path.join(directory, file), 'rb') as file:
        data = pickle.load(file)
        return data


def exists_checkpoint(file, directory=CHECKPOINTS_PATH):
    file = os.path.join(directory, file)
    return os.path.isfile(file)
