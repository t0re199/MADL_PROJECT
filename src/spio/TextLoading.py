import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Constants import TEXT_DATASET_PATH, RANDOM_STATE, CUSTOM_TEXT_DATASET_PATH


def load_text_dataset(path=TEXT_DATASET_PATH, size=0.1):
    dataframe_ = pd.read_excel(path)
    dataset_ = dataframe_.Text
    labels_ = dataframe_.Score

    X_train, X_test, y_train, y_test = train_test_split(dataset_,
                                                        labels_,
                                                        test_size=size,
                                                        random_state=RANDOM_STATE)

    dataset_ = X_test
    labels_ = y_test

    return dataset_, labels_


def load_full_text_dataset(path=TEXT_DATASET_PATH):
    dataframe_ = pd.read_excel(path)
    dataset_ = dataframe_.Text
    labels_ = dataframe_.Score
    return dataset_, labels_


def build_custom_dataset():
    dataset, labels = load_text_dataset()

    dataset = dataset.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    labels = np.where(labels == 0x1, 0x0, labels)
    labels = np.where(labels == 0x2, 0x0, labels)

    labels = np.where(labels == 0x3, 0x1, labels)

    labels = np.where(labels == 0x4, 0x2, labels)
    labels = np.where(labels == 0x5, 0x2, labels)

    return pd.DataFrame({"Text": dataset, "Score": pd.Series(labels)})


def load_validation_data(path):
    dataset, labels = load_full_text_dataset(path)

    dataset = dataset.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    labels = np.where(labels == 0x1, 0x0, labels)
    labels = np.where(labels == 0x2, 0x0, labels)

    labels = np.where(labels == 0x3, 0x1, labels)

    labels = np.where(labels == 0x4, 0x2, labels)
    labels = np.where(labels == 0x5, 0x2, labels)

    return pd.DataFrame({"Text": dataset, "Score": pd.Series(labels)})

def load_custom_dataset(path=CUSTOM_TEXT_DATASET_PATH):
    return pd.read_pickle(path)


if __name__ == '__main__':
    df = build_custom_dataset()
    df.to_csv("/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/text_dataset.csv")
