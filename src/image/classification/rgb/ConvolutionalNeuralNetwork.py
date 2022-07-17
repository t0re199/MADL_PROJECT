import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split, KFold

from Constants import RANDOM_STATE, RGB_IMAGE_SHAPE, FOLDS, IMAGE_CLASSES
from concurrency.KerasModelFitPerformer import KerasModelFitPerformer
from image.ImagePreProcessing import labels_to_int, normalize_dataset
from plot.Plots import plot_history, plot_cross_validation
from spio.ImageLoading import load_image_dataset
from spio.Models import *

tf.random.set_seed(RANDOM_STATE)

HISTORY_FILE = "cnn_rgb.hsvd"
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, "cnn_rgb.svd")
CROSS_VALIDATION = False

dataset, labels = load_image_dataset()

labels = labels_to_int(labels)
dataset = normalize_dataset(dataset)


def create_model():
    convolutional_nn_ = keras.Sequential()
    convolutional_nn_.add(layers.Conv2D(6, (7, 7),
                          activation='relu',
                          padding='same',
                          input_shape=RGB_IMAGE_SHAPE))
    convolutional_nn_.add(layers.BatchNormalization())

    convolutional_nn_.add(layers.AvgPool2D((2, 2),
                                           strides=2,
                                           padding="same")) # 40x30

    convolutional_nn_.add(layers.Conv2D(32, (5, 5),
                          activation='relu',
                          padding='same'))
    convolutional_nn_.add(layers.BatchNormalization())

    convolutional_nn_.add(layers.AvgPool2D((2, 2),
                                           strides=2,
                                           padding="same"))  # 20x15

    convolutional_nn_.add(layers.Conv2D(64, (3, 3),
                                        activation='relu',
                                        padding='same'))
    convolutional_nn_.add(layers.BatchNormalization())

    convolutional_nn_.add(layers.AvgPool2D((2, 2),
                                           strides=2,
                                           padding="same"))  # 10x7

    convolutional_nn_.add(layers.Conv2D(128, (3, 3),
                                        activation='relu',
                                        padding='same'))
    convolutional_nn_.add(layers.BatchNormalization())

    convolutional_nn_.add(layers.MaxPooling2D((2, 2),
                          strides=2,
                          padding="valid")) # 5x3

    convolutional_nn_.add(layers.Conv2D(512, (2, 2),
                                        activation='relu',
                                        padding='valid'))
    convolutional_nn_.add(layers.BatchNormalization())

    convolutional_nn_.add(layers.Flatten())
    convolutional_nn_.add(layers.Dropout(rate=0.55, seed=RANDOM_STATE))
    convolutional_nn_.add(layers.Dense(1000))
    convolutional_nn_.add(layers.Dropout(rate=0.5, seed=RANDOM_STATE))
    convolutional_nn_.add(layers.Dense(16))
    convolutional_nn_.add(layers.Dense(IMAGE_CLASSES, activation='softmax'))
    return convolutional_nn_


if CROSS_VALIDATION:
    k_fold = KFold(n_splits=FOLDS)

    scores = np.zeros(FOLDS, dtype=np.float64)

    performers = []

    fold = 0x0
    for training_indexes, test_indexes in k_fold.split(dataset):
        X_train = dataset[training_indexes]
        X_test = dataset[test_indexes]
        y_train = labels[training_indexes]
        y_test = labels[test_indexes]

        convolutional_nn = create_model()

        convolutional_nn.compile(loss="sparse_categorical_crossentropy",
                                 optimizer="adam",
                                 metrics=["accuracy"])

        performer = KerasModelFitPerformer(convolutional_nn,
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

    plot_cross_validation(scores, "CNN ACCURACY RGB")

else:
    X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.20,
                                                        random_state=RANDOM_STATE)

    convolutional_nn = create_model()
    convolutional_nn.summary()

    model_checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_FILE,
                                                       monitor="val_accuracy",
                                                       save_best_only=True,
                                                       mode="max")

    convolutional_nn.compile(loss="sparse_categorical_crossentropy",
                             optimizer="adam",
                             metrics=["accuracy"])

    if exists_checkpoint(HISTORY_FILE):
        convolutional_nn.load_weights(CHECKPOINT_FILE)
        history_dict = load_object(HISTORY_FILE)
    else:
        history_object = convolutional_nn.fit(X_train,
                                              y_train,
                                              epochs=10,
                                              validation_data=(X_test, y_test),
                                              callbacks=[model_checkpoint]
                                              )
        history_dict = history_object.history
        convolutional_nn.save_weights(CHECKPOINT_FILE)
        save_object(history_object.history, HISTORY_FILE)

    plot_history(history_dict, "CNN ACCURACY RGB")
