import os

import numpy as np
from PIL import Image

from Constants import *
from image.ImagePreProcessing import add_padding, reply_channel
from spio.Files import mark_to_skip, write_image_to_file


def load_image_dataset(path=IMAGE_DIRECTORY_PATH, flatten=False):
    if flatten:
        dataset_ = np.full(shape=(IMAGE_DATASET_SIZE,
                                  IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS),
                           fill_value=0x0,
                           dtype="float64")
    else:
        dataset_ = np.full(shape=(IMAGE_DATASET_SIZE,
                                  IMAGE_HEIGHT,
                                  IMAGE_WIDTH,
                                  IMAGE_CHANNELS),
                           fill_value=0x0,
                           dtype='float64')
    labels_ = []
    i = 0x0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".jpg"):
            image_ = np.asarray(Image.open(file_path, "r"))
            if flatten:
                dataset_[i] = image_.flatten()
            else:
                dataset_[i] = image_
            labels_.append(file.split("_")[0x0])
            i += 0x1
    return np.asarray(dataset_), np.array(labels_)


def raw_load_image_dataset(path=IMAGE_DIRECTORY_PATH):
    dataset_ = []
    labels_ = []
    invalid_images_indexes = []
    i = 0x0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".jpg"):
            image_ = np.asarray(Image.open(file_path, "r"))
            dataset_.append(image_)
            labels_.append(file.split("_")[0x0])
            if image_.shape != RGB_IMAGE_SHAPE:
                invalid_images_indexes.append(i)
            i += 1
    return np.asarray(dataset_), np.array(labels_), invalid_images_indexes


def load_grayscale_image_dataset(path=IMAGE_DIRECTORY_PATH, flatten=True):
    if flatten:
        dataset_ = np.full(shape=(IMAGE_DATASET_SIZE,
                                  IMAGE_HEIGHT * IMAGE_WIDTH),
                           fill_value=0x0,
                           dtype='float64')
    else:
        dataset_ = np.full(shape=(IMAGE_DATASET_SIZE,
                                  IMAGE_HEIGHT,
                                  IMAGE_WIDTH),
                           fill_value=0x0,
                           dtype='float64')
    labels_ = []
    i = 0x0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".jpg"):
            if flatten:
                image_ = np.asarray(Image.open(file_path, "r").convert("L"))
                dataset_[i] = image_.flatten()
            else:
                dataset_[i] = np.asarray(Image.open(file_path, "r").convert("L"))
            labels_.append(file.split("_")[0x0])
            i += 0x1
    return np.asarray(dataset_), np.array(labels_)


def check_image_dataset():
    dataset, labels, invalid_indexes = raw_load_image_dataset()

    shape_dict = dict()
    labels_dict = dict()
    for index in invalid_indexes:
        image = dataset[index]

        label = labels[index]
        label_amount = labels_dict.get(label)

        if label_amount is None:
            label_amount = 0x1
        else:
            label_amount += 0x1

        labels_dict[label] = label_amount

        shape = image.shape
        shape_amount = shape_dict.get(shape)

        if shape_amount is None:
            shape_amount = 0x1
        else:
            shape_amount += 0x1

        shape_dict[shape] = shape_amount

    print(shape_dict)
    print(labels_dict)


def fix_image_dataset(directory=IMAGE_DIRECTORY_PATH):
    dataset_ = np.full(shape=(IMAGE_DATASET_SIZE,
                              IMAGE_HEIGHT,
                              IMAGE_WIDTH,
                              IMAGE_CHANNELS),
                       fill_value=0x0)
    labels_ = []
    i = 0x0
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".jpg"):
            image_ = np.asarray(Image.open(file_path, "r"))
            if image_.shape == RGB_IMAGE_SHAPE:
                dataset_[i] = image_
            elif image_.shape == GRAYSCALE_IMAGE_SHAPE:
                image = reply_channel(image_)
                dataset_[i] = image
                mark_to_skip(file, directory)
                write_image_to_file(image, file, directory)
            else:
                image = add_padding(image_)
                dataset_[i] = image
                mark_to_skip(file, directory=directory)
                write_image_to_file(image, file, directory)

            labels_.append(file.split("_")[0x0])
            i += 0x1

    return np.asarray(dataset_), np.array(labels_)
