import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split, KFold

from Constants import RANDOM_STATE, FOLDS, EPOCHES, IMAGE_WIDTH, IMAGE_HEIGHT
from concurrency.KerasModelFitPerformer import KerasModelFitPerformer
from image.ImagePreProcessing import labels_to_int, normalize_dataset
from plot.Plots import plot_history, plot_cross_validation
from spio.ImageLoading import load_grayscale_image_dataset
from spio.Models import *

tf.random.set_seed(RANDOM_STATE)

HISTORY_FILE = "cnn_gs.hsvd"
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, "cnn_gs.svd")
CROSS_VALIDATION = False

dataset, labels = load_grayscale_image_dataset(flatten=False)

shape = dataset.shape
dataset = dataset.reshape(shape[0x0], shape[0x1], shape[0x2], 0x1)

labels = labels_to_int(labels)
dataset = normalize_dataset(dataset)


def create_model():
    cnn_gs_ = keras.Sequential()
    cnn_gs_.add(layers.Conv2D(64, (3, 3),
                              activation='relu',
                              padding='valid',
                              input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 0x1)))

    cnn_gs_.add(layers.MaxPooling2D((2, 2),
                                    padding="valid"))

    cnn_gs_.add(layers.Conv2D(64, (3, 3),
                              activation='relu',
                              padding='valid'))

    cnn_gs_.add(layers.MaxPooling2D((2, 2),
                                    padding="valid"))

    cnn_gs_.add(layers.Conv2D(64, (3, 3),
                              activation='relu',
                              padding='valid'))

    cnn_gs_.add(layers.MaxPooling2D((2, 2),
                                    padding="valid"))

    cnn_gs_.add(layers.Flatten())
    cnn_gs_.add(layers.Dropout(rate=0.45, seed=RANDOM_STATE))
    cnn_gs_.add(layers.Dense(512, activation='relu'))
    cnn_gs_.add(layers.Dense(256, activation='relu'))
    cnn_gs_.add(layers.Dense(128, activation='relu'))
    cnn_gs_.add(layers.Dropout(rate=0.23, seed=RANDOM_STATE))
    cnn_gs_.add(layers.Dense(32, activation='relu'))
    cnn_gs_.add(layers.Dense(16, activation='relu'))
    cnn_gs_.add(layers.Dense(4, activation='softmax'))
    return cnn_gs_


if CROSS_VALIDATION:
    k_fold = KFold(n_splits=FOLDS)

    scores = np.zeros(FOLDS, dtype="float64")

    performers = []

    fold = 0x0

    for training_indexes, test_indexes in k_fold.split(dataset):
        X_train = dataset[training_indexes]
        X_test = dataset[test_indexes]
        y_train = labels[training_indexes]
        y_test = labels[test_indexes]

        cnn_gs = create_model()

        cnn_gs.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])

        performer = KerasModelFitPerformer(cnn_gs,
                                           scores,
                                           fold,
                                           X_train,
                                           X_test,
                                           y_train,
                                           y_test)
        performer.start()
        performers.append(performer)
        fold += 0x1

        for performer in performers:
            performer.join()

    plot_cross_validation(scores, "CNN " + str(EPOCHES) + " ACCURACY GS")

else:
    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.30,
                                                        random_state=RANDOM_STATE)

    cnn_gs = create_model()
    cnn_gs.summary()

    cnn_gs.compile(loss="sparse_categorical_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])

    if False and exists_checkpoint(HISTORY_FILE):
        cnn_gs.load_weights(CHECKPOINT_FILE)
        history_dict = load_object(HISTORY_FILE)
    else:
        history_object = cnn_gs.fit(X_train,
                                    y_train,
                                    epochs=EPOCHES,
                                    validation_data=(X_test, y_test))
        history_dict = history_object.history
        cnn_gs.save_weights(CHECKPOINT_FILE)
        save_object(history_object.history, HISTORY_FILE)

    plot_history(history_dict, "CNN " + str(EPOCHES) + " ACCURACY GS")
