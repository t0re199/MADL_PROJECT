import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from image.classification.rgb.PCA import load_rgb_pca_dataset
from plot.Plots import plot_cross_validation
from spio.Models import save_object, load_object, exists_checkpoint

CHECKPOINT_FILE = "svm_pca.svd"
CROSS_VALIDATION = False

dataset, labels = load_rgb_pca_dataset()

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

if CROSS_VALIDATION:
    support_vector_classifier = SVC(decision_function_shape="ovo",
                                    C=0x64,
                                    gamma=0.001,
                                    kernel="rbf")

    scores = cross_val_score(support_vector_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "PCA Support Vector Machine Score")
else:
    if exists_checkpoint(CHECKPOINT_FILE):
        support_vector_classifier = load_object(CHECKPOINT_FILE)
    else:
        param_grid = [
            {"C": [1, 10, 100], "kernel": ["linear", "rbf"]},
            {"C": [1, 10, 100], "gamma": [0.001, 0.0001], "kernel": ["rbf"]}
        ]

        support_vector_classifier = SVC(decision_function_shape="ovo",
                                        C=0x64,
                                        gamma=0.001,
                                        kernel="rbf")

        grid_search_cv = GridSearchCV(support_vector_classifier,
                                      param_grid,
                                      n_jobs=-0x1,
                                      verbose=0x1)

        grid_search_cv.fit(X_train, y_train)
        """
            grid_search_cv.best_params_ -> {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
        """
        support_vector_classifier = grid_search_cv.best_estimator_
        save_object(support_vector_classifier, CHECKPOINT_FILE)
        y_pred = support_vector_classifier.predict(X_test)

    disp = plot_confusion_matrix(support_vector_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("PCA Support Vector Machine Confusion Matrix RGB")
    plt.show()
