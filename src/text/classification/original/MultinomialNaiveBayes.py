import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from plot.Plots import plot_cross_validation
from spio.Models import load_object, exists_checkpoint
from spio.TextLoading import load_text_dataset
from text.TextPreProcessing import vectorize_dataset, preprocess_dataset

CHECKPOINT_FILE = "text_cnb_5_classes.svd"
CROSS_VALIDATION = False

dataset, labels = load_text_dataset(size=0.05)

dataset = preprocess_dataset(dataset)
dataset = vectorize_dataset(dataset).toarray()

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

if CROSS_VALIDATION:
    mnb_classifier = MultinomialNB()

    scores = cross_val_score(mnb_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "Multinomial Naive Bayes Score GS")
else:
    if exists_checkpoint(CHECKPOINT_FILE):
        mnb_classifier = load_object(CHECKPOINT_FILE)
    else:

        mnb_classifier = MultinomialNB()

        mnb_classifier.fit(X_train, y_train)

        #save_object(cnb_classifier, CHECKPOINT_FILE)
        y_pred = mnb_classifier.predict(X_test)

    disp = plot_confusion_matrix(mnb_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("Multinomial Naive Bayes Confusion Matrix")
    plt.show()
