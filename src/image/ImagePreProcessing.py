import numpy as np

from Constants import *
from PIL import Image


LABELS_DICT = {
    "handbags": 0x0,
    "sports-shoes": 0x1,
    "tops": 0x2,
    "trousers": 0x3
}


def labels_to_int(labels):
    return np.array([LABELS_DICT.get(item) for item in labels])


def normalize_dataset(dataset):
    return dataset / float(0xff)


def labels_to_anomaly_detection(labels):
    anomaly_detection_labels = np.ones_like(labels, dtype=np.int)
    anomaly_detection_labels[np.argwhere(labels != "sports-shoes")] = -0x1

    return anomaly_detection_labels


def reply_channel(image):
    result = np.full(RGB_IMAGE_SHAPE,
                     fill_value=0xff)

    for i in range(IMAGE_CHANNELS):
        result[:, :, i] = image

    return result


def add_padding(image):
    result = np.full(RGB_IMAGE_SHAPE,
                     fill_value=0xff)

    width = image.shape[0x1]
    height = image.shape[0x0]

    filling_values = []
    for channel in range(IMAGE_CHANNELS):
        filling_values.append(image[height - 0x1, width - 0x1, channel])

    y_value = IMAGE_HEIGHT - height
    y_offset = y_value // 0x2

    x_value = IMAGE_WIDTH - width
    x_offset = x_value // 0x2

    for i in range(y_offset):
        for channel in range(IMAGE_CHANNELS):
            result[i, :, channel] = filling_values[channel]
            result[IMAGE_HEIGHT - 0x1 - i, :, channel] = filling_values[channel]

    for j in range(x_offset):
        for channel in range(IMAGE_CHANNELS):
            result[:, j, channel] = filling_values[channel]
            result[:, IMAGE_WIDTH - 0x1 - j, channel] = filling_values[channel]

    for i in range(height):
        for j in range(width):
            for channel in range(IMAGE_CHANNELS):
                result[i + y_offset, j + x_offset, channel] = image[i, j, channel]

    return result


def squarefy_rgb(dataset):
    shape = (dataset.shape[0], dataset.shape[1], dataset.shape[1], 3)
    squared_ds = np.zeros(shape, dtype=np.uint8)
    dataset = dataset.astype(np.uint8)

    for image in range(dataset.shape[0]):
        squared_ds[image] = np.asarray(Image.fromarray(dataset[image]).resize((80, 80)))
    return squared_ds
