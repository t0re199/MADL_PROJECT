import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from image.ImagePreProcessing import labels_to_int, normalize_dataset
from plot.Plots import plot_cross_validation
from spio.ImageLoading import load_grayscale_image_dataset
from spio.Models import load_object, save_object, exists_checkpoint

CHECKPOINT_FILE = "cnb.svd"
CROSS_VALIDATION = True

dataset, labels = load_grayscale_image_dataset()

labels = labels_to_int(labels)
dataset = normalize_dataset(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

if CROSS_VALIDATION:
    cnb_classifier = CategoricalNB()

    scores = cross_val_score(cnb_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "Categorical Naive Bayes Score GS")
else:
    if exists_checkpoint(CHECKPOINT_FILE):
        cnb_classifier = load_object(CHECKPOINT_FILE)
    else:

        cnb_classifier = CategoricalNB()

        cnb_classifier.fit(X_train, y_train)

        save_object(cnb_classifier, CHECKPOINT_FILE)
        y_pred = cnb_classifier.predict(X_test)

    disp = plot_confusion_matrix(cnb_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("Categorical Naive Bayes Confusion Matrix GS")
    plt.show()
