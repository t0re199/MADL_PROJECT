import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping

from Constants import RANDOM_STATE, MAX_LEN, FOLDS
from plot.Plots import plot_history, plot_cross_validation
from spio.Models import *
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset

HISTORY_FILE = "text_snn.hsvd"
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, "text_snn.svd")
CROSS_VALIDATION = True

dataframe = load_custom_dataset()

dataset = preprocess_dataset(np.copy(dataframe.Text.values))
labels = np.copy(dataframe.Score.values)

tokenizer = Tokenizer(num_words=MAX_LEN)
tokenizer.fit_on_texts(dataset)

dataset = tokenizer.texts_to_sequences(dataset)
corpus_len = len(tokenizer.word_index) + 0x1

dataset = pad_sequences(dataset, padding="post", maxlen=MAX_LEN)


def create_model(corpus_len, max_len=MAX_LEN):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=corpus_len,
                               output_dim=16,
                               input_length=max_len))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(3, activation='softmax'))
    return model


if CROSS_VALIDATION:
    k_fold = KFold(n_splits=FOLDS)

    scores = np.zeros(FOLDS, dtype="float64")

    fold = 0x0

    for training_indexes, test_indexes in k_fold.split(dataset):
        print(f"Fold {fold}")
        X_train = dataset[training_indexes]
        X_test = dataset[test_indexes]
        y_train = labels[training_indexes]
        y_test = labels[test_indexes]

        early_stopping = EarlyStopping(restore_best_weights=True,
                                       monitor='val_accuracy',
                                       mode="max",
                                       patience=2)

        model = create_model(corpus_len)

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        history_object = model.fit(X_train,
                                   y_train,
                                   epochs=60,
                                   callbacks=[early_stopping],
                                   validation_data=(X_test, y_test))

        history_dict = history_object.history
        scores[fold] = np.max(history_object.history["val_accuracy"])

        fold += 0x1

    plot_cross_validation(scores, "SEQUENTIAL NEURAL NETWORK SCORE")

else:
    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.30,
                                                        random_state=RANDOM_STATE)

    model = create_model(corpus_len)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    if exists_checkpoint(HISTORY_FILE):
        model.load_weights(CHECKPOINT_FILE)
        history_dict = load_object(HISTORY_FILE)
    else:
        early_stopping = EarlyStopping(restore_best_weights=True,
                                       monitor='val_accuracy',
                                       mode="max",
                                       patience=3)

        history_object = model.fit(X_train,
                                   y_train,
                                   epochs=60,
                                   callbacks=[early_stopping],
                                   validation_data=(X_test, y_test))

        history_dict = history_object.history
        model.save_weights(CHECKPOINT_FILE)
        save_object(history_object.history, HISTORY_FILE)

    plot_history(history_dict, "SEQUENTIAL NEURAL NETWORK ACCURACY")
