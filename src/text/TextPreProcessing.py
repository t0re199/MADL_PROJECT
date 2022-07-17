import re

import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
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
