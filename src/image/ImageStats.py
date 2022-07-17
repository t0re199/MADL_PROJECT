import matplotlib.pyplot as plt
import numpy as np

from image.ImagePreProcessing import labels_to_anomaly_detection
from spio.ImageLoading import load_image_dataset


def plot_class_hist():
    dataset, labels = load_image_dataset()

    items_dict = {}

    for label in labels:
        amount = (labels == label).sum()
        items_dict[label] = amount

    print(items_dict)
    colors = ["cyan", "lighblue", "steelblue", "cornflowerblue"]
    plt.set_cmap("Blues")
    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 100, 100))
    plt.show()


def plot_malformed_hist():
    shapes=["(80, 60)", "(80, 53, 3)", "(60, 60, 3)", "(75, 60, 3)"]
    amount = [38, 1, 1, 1]

    plt.bar(shapes, height=amount)
    plt.yticks(np.arange(0, max(amount)+10, 10))
    plt.show()


def plot_anomaly_hist():

    names = {"1": "inliers (1)",
             "-1": "outliers (-1)"}
    dataset, labels = load_image_dataset()

    labels = labels_to_anomaly_detection(labels)
    items_dict = {}

    for label in np.unique(labels):
        amount = (labels == label).sum()
        items_dict[names[str(label)]] = amount

    print(items_dict)
    colors = ["cyan", "lighblue", "steelblue", "cornflowerblue"]
    plt.set_cmap("Blues")
    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 500, 500))
    plt.show()



if __name__ == '__main__':
    dataset, labels = load_image_dataset()

    labels = labels_to_anomaly_detection(labels)

    normal = dataset[np.argwhere(labels == 1)]
    anomalous = dataset[np.argwhere(labels == -1)]