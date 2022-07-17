import os
import sys
from optparse import OptionParser

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, roc_curve, auc

from Constants import RGB_IMAGE_SHAPE, RANDOM_STATE, IMAGE_CLASSES
from image.ImagePreProcessing import labels_to_int, normalize_dataset, labels_to_anomaly_detection
from spio.ImageLoading import load_image_dataset, load_grayscale_image_dataset
from spio.Models import load_object
from spio.TextLoading import load_validation_data
from text.TextPreProcessing import preprocess_dataset, vectorize_dataset

IMAGE = 0x0
TEXT = 0x1

images_checkpoint_dict = {"ada": "adaboost.svd",
                          "cnb": "cnb.svd",
                          "knn": "knn.svd",
                          "svm": "svm_rgb_ovo.svd",
                          "cnn": "checkpoints/cnn_rgb.svd",
                          "1csvm": "image_anomaly_svm.svd"
                          }

text_checkpoint_dict = {"ada": "text_adaboost.svd",
                        "mnb": "text_mnb.svd",
                        "knn": "text_knn.svd",
                        "svm": "text_svm.svd",
                        "snn": "checkpoints/text_snn.svd",
                        "1csvm": "anomaly_text_svm.svd"
                        }

image_models = set(list(images_checkpoint_dict.keys()))
text_models = set(list(text_checkpoint_dict.keys()))


def create_image_model():
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


def create_text_model(corpus_len, max_len):
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


def is_neural(model_):
    return model_ == "cnn" or model_ == "snn"


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--imageData", action="store_true", dest="image")
    parser.add_option("-t", "--textData", action="store_true", dest="text")
    parser.add_option("-a", "--anomalyDetection", action="store_true", dest="anomaly")
    parser.add_option("-l", "--listModels", action="store_true",
                      help="shows available models and exit",
                      dest="list_models")
    parser.add_option("-m", "--model", dest="model", type="string")
    parser.add_option("-P", "--validationPath", dest="validation_path", type="string")

    (options, args) = parser.parse_args()

    image_data = options.image if options.image is not None else False
    text_data = options.text if options.text is not None else False
    anomaly = options.anomaly if options.anomaly is not None else False

    if options.list_models:
        print("Images Classifiers:")
        for classifier in images_checkpoint_dict.keys():
            print("\t", classifier, end="")

        print("\nTexts Classifiers:")
        for classifier in text_checkpoint_dict.keys():
            print("\t", classifier, end="")
        print()
        exit(0)

    if image_data ^ text_data == 0x0:
        sys.stderr.write("Evaluate.py -h for help\n")
        exit(199)

    if options.model is None:
        sys.stderr.write("Invalid Model!\nEvaluate.py -h for help\n")
        exit(198)

    if image_data and options.model not in image_models:
        sys.stderr.write("Invalid Model!\nEvaluate.py -h for help\n")
        exit(198)

    if text_data and options.model not in text_models:
        sys.stderr.write("Invalid Model!\nEvaluate.py -h for help\n")
        exit(198)

    if options.validation_path is None:
        sys.stderr.write("Evaluate.py -h for help\n")
        exit(199)
    else:
        if options.image and not os.path.isdir(options.validation_path):
            sys.stderr.write("A directory was expected\n")
            exit(200)

        if options.text and (not os.path.isfile(options.validation_path) or not options.validation_path.endswith(".xlsx")):
            sys.stderr.write("An .xlxs file was expected\n")
            exit(201)

    model = options.model

    if options.image:

        checkpoint = images_checkpoint_dict.get(model)
        if anomaly:
            if checkpoint.find("anomaly") == -0x1:
                sys.stderr.write("You Entered A Model Not For Anomaly Detection!.")
                exit(202)

            dataset, labels = load_image_dataset(flatten=True)

            labels = labels_to_anomaly_detection(labels)
            dataset = normalize_dataset(dataset)


            support_vector_classifier = load_object(checkpoint)
            scores = support_vector_classifier.score_samples(dataset)

            false_positive_rate, true_positive_rate, threshold = roc_curve(labels, scores)
            auc_score = auc(false_positive_rate, true_positive_rate)

            print(f"AUC Score {auc_score}")

        else:
            if checkpoint.find("rgb") != -0x1:
                dataset, labels = load_image_dataset(path=options.validation_path, flatten=model == "svm")
            else:
                dataset, labels = load_grayscale_image_dataset(path=options.validation_path, flatten=True)
            labels = labels_to_int(labels)
            dataset = normalize_dataset(dataset)


            if is_neural(model):
                classifier = create_image_model()

                classifier.compile(loss="sparse_categorical_crossentropy",
                                   optimizer="adam",
                                   metrics=["accuracy"])

                classifier.load_weights(checkpoint)
                predicted = classifier.predict_classes(dataset)
                predicted = classifier.predict_classes(dataset)
            else:
                classifier = load_object(checkpoint)
                predicted = classifier.predict(dataset)

            score = accuracy_score(labels, predicted)
            print(f"Accuracy Score: {score}")

    else:
        checkpoint = text_checkpoint_dict.get(model)

        dataframe = load_validation_data(options.validation_path)
        dataset = preprocess_dataset(np.copy(dataframe.Text.values))
        labels = dataframe.Score.values

        if anomaly:
            if checkpoint.find("anomaly") == -0x1:
                sys.stderr.write("You Entered A Model Not For Anomaly Detection!.")
                exit(202)
            support_vector_classifier = load_object(checkpoint)
            scores = support_vector_classifier.score_samples(dataset)

            false_positive_rate, true_positive_rate, threshold = roc_curve(labels, scores)
            auc_score = auc(false_positive_rate, true_positive_rate)

            print(f"AUC Score {auc_score}")
        else:
            if is_neural(model):
                MAX_LEN = 3000

                tokenizer = Tokenizer(num_words=MAX_LEN)
                tokenizer.fit_on_texts(dataset)

                dataset = tokenizer.texts_to_sequences(dataset)
                corpus_len = len(tokenizer.word_index) + 0x1

                dataset = pad_sequences(dataset, padding="post", maxlen=MAX_LEN)

                model = create_text_model(corpus_len, MAX_LEN)

                model.compile(loss="sparse_categorical_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])

                model.load_weights(checkpoint)
                predicted = model.predict_classes(dataset)
            else:
                dataset = vectorize_dataset(dataset).toarray()

                classifier = load_object(checkpoint)
                predicted = classifier.predict(dataset)

            score = accuracy_score(labels, predicted)
            print(f"Accuracy Score: {score}")
