import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from spio.TextLoading import load_text_dataset, load_full_text_dataset, load_custom_dataset
from text.TextPreProcessing import labels_to_anomaly_detection


def plot_class_hist():
    dataset, labels = load_full_text_dataset()

    items_dict = {}

    for label in np.unique(labels):
        amount = (labels == label).sum()
        items_dict[label] = amount

    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 2000, 2000))
    plt.title("Class Hist")
    plt.show()


def plot_reduced_class_hist():
    dataset, labels = load_text_dataset()

    items_dict = {}

    for label in np.unique(labels):
        amount = (labels == label).sum()
        items_dict[label] = amount


    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 200, 200))
    plt.show()


def plot_custom_class_hist():
    dataframe = load_custom_dataset()

    dataset = dataframe.Text.values
    labels = np.array(dataframe.Score.values, dtype=np.int)

    items_dict = {}

    for label in np.unique(labels):
        amount = (labels == label).sum()
        items_dict[label] = amount

    print(items_dict)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 500, 500))
    plt.show()


def plot_anomaly_hist():

    names = {"1": "inliers (1)",
             "-1": "outliers (-1)"}
    dataframe = load_custom_dataset()

    dataset = dataframe.Text.values
    labels = labels_to_anomaly_detection(dataframe.Score.values)

    items_dict = {}

    for label in np.unique(labels):
        amount = (labels == label).sum()
        items_dict[names[str(label)]] = amount

    print(items_dict)

    bars = plt.bar(items_dict.keys(), height=items_dict.values())
    plt.yticks(np.arange(0, max(items_dict.values()) + 500, 500))
    plt.show()



if __name__ == '__main__':
    dataframe = load_custom_dataset()

    dataset = dataframe.Text.values
    #dataset = preprocess_dataset(dataset)

    # labels = np.array(dataframe.Score.values, dtype=np.int)
    #
    # corpuses = []
    # for label in range(3):
    #     ds = dataset[np.argwhere(labels == label)]
    #     ds = ds.reshape(ds.shape[0])
    #     corpus = get_corpus(ds)
    #     corpuses.append(corpus)
    #
    # int_0_1 = corpuses[0][0].intersection(corpuses[1][0])
    # int_1_2 = corpuses[1][0].intersection(corpuses[2][0])
