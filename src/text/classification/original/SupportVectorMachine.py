import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from plot.Plots import plot_cross_validation
from spio.Models import load_object, exists_checkpoint
from spio.TextLoading import load_text_dataset
from text.TextPreProcessing import preprocess_dataset, get_corpus, vectorize_dataset

CHECKPOINT_FILE = "text_full_svm_ovo.svd"
CROSS_VALIDATION = False

dataset, labels = load_text_dataset()

dataset = preprocess_dataset(np.copy(dataset.values))
labels = labels.values

corpus, max_len = get_corpus(dataset)

dataset = vectorize_dataset(dataset, corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)
if CROSS_VALIDATION:
    support_vector_classifier = SVC(decision_function_shape="ovo",
                                    C=0xa,
                                    kernel="rbf")

    scores = cross_val_score(support_vector_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "Support Vector Machine Score")
else:
    if exists_checkpoint(CHECKPOINT_FILE):
        support_vector_classifier = load_object(CHECKPOINT_FILE)
    else:
        param_grid = [
            {"C": [10, 100, 1000, 10000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]}
        ]

        support_vector_classifier = SVC(decision_function_shape="ovo",
                                        C=0xa,
                                        kernel="rbf")

        grid_search_cv = GridSearchCV(support_vector_classifier,
                                      param_grid,
                                      n_jobs=-0x1,
                                      verbose=0x1)

        #grid_search_cv.fit(X_train, y_train)
        support_vector_classifier.fit(X_train, y_train)

        """
            grid_search_cv.best_params_ -> {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
        """
        #support_vector_classifier = grid_search_cv.best_estimator_
        #save_object(support_vector_classifier, CHECKPOINT_FILE)
        y_pred = support_vector_classifier.predict(X_test)

    disp = plot_confusion_matrix(support_vector_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("Support Vector Machines Confusion Matrix")
    plt.show()
