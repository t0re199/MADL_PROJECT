import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from image.ImagePreProcessing import labels_to_int, normalize_dataset
from plot.Plots import plot_cross_validation
from spio.ImageLoading import load_grayscale_image_dataset
from spio.Models import load_object, save_object, exists_checkpoint

CHECKPOINT_FILE = "svm_c_10_df_ovo.svd"
CROSS_VALIDATION = False

dataset, labels = load_grayscale_image_dataset()

labels = labels_to_int(labels)
dataset = normalize_dataset(dataset)

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

    plot_cross_validation(scores, "Support Vector Machine Score GS")
else:
    if exists_checkpoint(CHECKPOINT_FILE):
        support_vector_classifier = load_object(CHECKPOINT_FILE)
    else:
        param_grid = [
            {"C": [1, 10, 100, 1000], "kernel": ["poly", "rbf"]},
            {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]}
        ]

        support_vector_classifier = SVC(decision_function_shape="ovo",
                                        C=0x64,
                                        gamma=0.001,
                                        kernel="rbf")

        grid_search_cv = GridSearchCV(support_vector_classifier,
                                      param_grid,
                                      n_jobs=-0x1,
                                      verbose=0x1)

        #grid_search_cv.fit(X_train, y_train)
        support_vector_classifier.fit(X_train, y_train)
        """
            grid_search_cv.best_params_ -> {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
        """
        #support_vector_classifier = grid_search_cv.best_estimator_
        save_object(support_vector_classifier, CHECKPOINT_FILE)
        y_pred = support_vector_classifier.predict(X_test)

    disp = plot_confusion_matrix(support_vector_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("Support Vector Machine Confusion Matrix GS")
    plt.show()
