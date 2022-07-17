import re
import re

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Constants import TEXT_DATASET_PATH, RANDOM_STATE


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


def get_wordnet_pos(tag):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag[0], wordnet.NOUN)


def vectorize_dataset(dataset_):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=0x7, max_df=0.8, stop_words=stopwords.words('english'))
    return tfidf_vectorizer.fit_transform(dataset_)


def lemmatize_dataset(dataset_):
    regexp_tokenizer = RegexpTokenizer(r'\w+')
    word_net_lemmatizer = WordNetLemmatizer()
    for i in range(dataset_.size):
        dataset_[i] = " ".join([word_net_lemmatizer.lemmatize(word) for word in regexp_tokenizer.tokenize(dataset_[i])])
    return dataset_


def clean_dataset(dataframe_):
    return dataframe_.apply(lambda text: re.sub(r'\W+', ' ', text.strip().lower()))


if __name__ == '__main__':

    dataframe_ = pd.read_excel(TEXT_DATASET_PATH)
    dataset_ = dataframe_.Text
    labels_ = dataframe_.Score

    X_train, X_test, y_train, y_test = train_test_split(dataset_,
                                                        labels_,
                                                        test_size=0.1,
                                                        random_state=RANDOM_STATE)

    dataset_ = X_test
    labels_ = y_test

    print("Before:")

    for i in range(1, 6):
        indexes = np.argwhere(labels_[labels_ == i].values == i)
        print(i, " ", indexes.size)

    for i in [2,4]:
        indexes = y_train[y_train == i].index[:1500]
        dataset_ = pd.concat([dataset_, X_train[indexes]])
        labels_ = pd.concat([labels_, y_train[indexes]])

    print("\nAfter")
    for i in range(1, 6):
        indexes = np.argwhere(labels_[labels_ == i].values == i)
        print(i, " ", indexes.size)


    dataset_ = dataset_.reset_index(drop=True)
    labels = labels_.values
    dataset = clean_dataset(dataset_)

    dataset = dataset.values

    ##LEMMATIZATION

    dataset = lemmatize_dataset(dataset)

    ###END LEMMATIZATION


    ##CORPUS

    regexp_tokenizer = RegexpTokenizer(r'\w+')
    corpus = set()

    max_len = 0x0

    for text in dataset:
        max_len = max(max_len, len(text))
        corpus = corpus.union(regexp_tokenizer.tokenize(text))

    #####END CORPUS



    ##VECOTORIZATION

    tfidf_vectorizer = TfidfVectorizer(max_features=1500,
                                       vocabulary=corpus)

    ml_dataset = tfidf_vectorizer.fit_transform(dataset).toarray()
    dataset = ml_dataset


    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.30,
                                                        random_state=RANDOM_STATE)

    model = keras.Sequential()
    model.add(layers.LSTM(256, input_shape=X_train.shape, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])

    history_object = model.fit(X_train,
                                y_train,
                                epochs=10,
                                validation_data=(X_test, y_test))


