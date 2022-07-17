import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')



"""
/////////////////////////////////////////////////////////////////PATH///////////////////////////////////////////////////
"""
IMAGE_DIRECTORY_PATH = ""
CHECKPOINTS_PATH = ""

SKIP_EXTENSION = ".skip"

TEXT_DATASET_PATH = ""
CUSTOM_TEXT_DATASET_PATH = "drive/MyDrive/dataset/text_dataset.ds4"
"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


"""
/////////////////////////////////////////////////////////////EXPERIMENTS////////////////////////////////////////////////
"""

RANDOM_STATE = 0xd

COLOR_MAP = "Blues"

FOLDS = 0xa

EPOCHES = 0xf

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


"""
////////////////////////////////////////////////////////////IMAGE DATASET///////////////////////////////////////////////
"""

IMAGE_CLASSES = 0x4

IMAGE_DATASET_SIZE = 5477

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 60
IMAGE_CHANNELS = 3

RGB_IMAGE_SHAPE = (80, 60, 3)
GRAYSCALE_IMAGE_SHAPE = (80, 60)

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


"""
////////////////////////////////////////////////////////////TEXT DATASET///////////////////////////////////////////////
"""

CUSTOM_TEXT_LABELS = [0x0, 0x1, 0x2]

MAX_LEN = 5000

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

import pandas as pd
from sklearn.model_selection import train_test_split



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


def load_custom_dataset(path=CUSTOM_TEXT_DATASET_PATH):
    return pd.read_pickle(path)


import re
import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_wordnet_pos(tag):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag[0], wordnet.NOUN)


def vectorize_dataset(dataset, max_features=2000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features,
                                       max_df=0.5,
                                       min_df=1,
                                       ngram_range=(1, 2),
                                       norm="l2",
                                       smooth_idf=True)

    return tfidf_vectorizer.fit_transform(dataset)


def preprocess_dataset(dataset):
    word_net_lemmatizer = WordNetLemmatizer()

    en_stopwords = set(stopwords.words("english"))

    for i in range(dataset.shape[0]):
        dataset[i] = re.sub(r'\W+', ' ', dataset[i]).strip().lower()

        tagged_words = pos_tag(nltk.word_tokenize(dataset[i]))

        processed_words = []

        for j, (word, tag) in enumerate(tagged_words):
            if word == 't' and j > 0x0 and tagged_words[j - 0x1][0x1][0x0] == 'V':
                if len(processed_words) > 0x0:

                    processed_words[-0x1] = word_net_lemmatizer.lemmatize(tagged_words[j - 0x1][0x0][: -0x1],
                                                                          get_wordnet_pos('V'))
                    processed_words.append("not")
                else:
                    processed_words.append("not")
                    processed_words.append(word_net_lemmatizer.lemmatize(tagged_words[j - 0x1][0x0][: -0x1],
                                                                         get_wordnet_pos('V')))

            elif word == 'm' and j > 0x0 and tagged_words[j - 0x1][0x1][0x0] == 'N':
                if len(processed_words) > 0x0:
                    processed_words[-0x1] = "be"
                else:
                    processed_words.append("be")

            elif len(word) > 0x1 and word not in en_stopwords and word.isalpha():
                processed_words.append(word_net_lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

        dataset[i] = " ".join(processed_words)

    return dataset


def get_corpus(dataset, clean=False):
    corpus = set()
    max_len = 0x0

    if not clean:
        dataset = preprocess_dataset(dataset)

    for text in dataset:
        tokens = nltk.word_tokenize(text)
        corpus = corpus.union(tokens)
        max_len = max(max_len, len(tokens))

    return corpus, max_len


def labels_to_anomaly_detection(labels):
    return np.where(labels == 0x2, 0x1, -0x1)



import matplotlib.pyplot as plt


def plot_history(history_dict, title, metric="accuracy"):
    acc = history_dict[metric]
    val_acc = history_dict['val_'+metric]
    epochs = range(0x1, len(acc) + 0x1)
    plt.figure()
    plt.plot(epochs, acc, "bo", label="Training " + metric.capitalize())
    plt.plot(epochs, val_acc, "b", label="Validation " + metric.capitalize())
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend(loc="lower right")
    plt.ion()
    plt.show()


def plot_cross_validation(scores, title, metric="accuracy"):
    metric = metric.capitalize()
    folds = range(0x1, len(scores) + 0x1)
    plt.figure()
    plt.plot(folds, scores, "b", label=metric)
    plt.title(title)
    plt.xlabel("Folds")
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.ion()
    plt.show()


def plot_roc_curve(true_positive_rate, false_positive_rate, auc_score):
    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % auc_score)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
