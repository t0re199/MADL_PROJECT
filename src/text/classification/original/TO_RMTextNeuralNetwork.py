import re
import re

import nltk
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from nltk import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

LOAD_FROM_DISK = True

"""
/////////////////////////////////////////////////////////////////PATH///////////////////////////////////////////////////
"""
IMAGE_DIRECTORY_PATH = "/Users/t0re199/Documents/Uni/LM/Machine e Deep Learning/Progetto/immagini-3"
CHECKPOINTS_PATH = "/checkpoints"

SKIP_EXTENSION = ".skip"

TEXT_DATASET_PATH = "/Users/t0re199/Documents/Uni/LM/Machine e Deep Learning/Progetto/testi-4.xlsx"
TEXT_PREPROCESSED_DATASET = "/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/"
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


def load_text_dataset(path=TEXT_DATASET_PATH, size=0.1):
    dataframe_ = pd.read_excel(path)
    dataset_ = dataframe_.Text
    labels_ = dataframe_.Score.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(dataset_,
                                                        labels_,
                                                        test_size=size,
                                                        random_state=RANDOM_STATE)

    dataset_ = X_test
    labels_ = y_test

    return dataset_, labels_

TAG_DICT__ = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

def get_wordnet_pos(tag):


    return TAG_DICT__.get(tag[0], wordnet.NOUN)


def vectorize_dataset(dataset_, corpus, max_features=5000):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features,
                                        vocabulary=corpus,
                                        stop_words=stopwords.words('english'))
    return tfidf_vectorizer.fit_transform(dataset_)


def lemmatize_dataset(dataframe_):
    word_net_lemmatizer = WordNetLemmatizer()
    for i in range(dataframe_.size):
        tagged_words = pos_tag(nltk.word_tokenize(dataframe_[i]))
        dataframe_[i] = " ".join([word_net_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words])
    return dataframe_


def clean_dataset(dataframe_):
    return dataframe_.apply(lambda text: re.sub(r'\W+', ' ',
                            text.strip().lower()))

if not LOAD_FROM_DISK:
    dataframe = pd.read_excel(TEXT_DATASET_PATH)

    dataset = dataframe.Text
    labels = dataframe.Score

    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                         labels,
                                                         test_size=0.1,
                                                         random_state=RANDOM_STATE)

    dataset = X_test
    labels = y_test

    X_train = None
    y_train = None


    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                         labels,
                                                         test_size=0.3,
                                                         random_state=RANDOM_STATE)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)


    X_train = clean_dataset(X_train)
    X_test = clean_dataset(X_test)


    X_train = lemmatize_dataset(X_train)
    X_test = lemmatize_dataset(X_test)



    y_train = y_train.values - 0x1
    y_test = y_test.values - 0x1


if LOAD_FROM_DISK:
    X_train = np.load("/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/lemma_train.npy",
                      allow_pickle=True)

    X_test = np.load("/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/lemma_test.npy",
                     allow_pickle=True)

    y_train = np.load("/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/lemma_train_labels.npy",
                      allow_pickle=True)

    y_test = np.load("/Users/t0re199/git/mldl_project/MLDL_PROJECT/datasets/lemma_test_labels.npy",
                     allow_pickle=True)

else:
    np.save("/datasets/lemma_train.npy",
            X_train, allow_pickle=True)

    np.save("/datasets/lemma_train_labels.npy",
            y_train, allow_pickle=True)


    np.save("/datasets/lemma_test.npy",
            X_test, allow_pickle=True)

    np.save("/datasets/lemma_test_labels.npy",
            y_test, allow_pickle=True)





tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
corpus_len = len(tokenizer.word_index) + 0x1 # 0x0 reserved for padding

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_test)
X_test = tokenizer.texts_to_sequences(X_test)


X_train = pad_sequences(X_train, padding='post', maxlen=5000)
X_test = pad_sequences(X_test, padding='post', maxlen=5000)


model = keras.Sequential()
model.add(layers.Embedding(input_dim=corpus_len,
                           output_dim=64,
                           input_length=5000))

model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(256))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history_object = model.fit(X_train,
                           y_train,
                           verbose=True,
                           validation_data=(X_test, y_test),
                           epochs=0x14)