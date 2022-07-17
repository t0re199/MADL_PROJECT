import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from image.ImagePreProcessing import labels_to_int, normalize_dataset
from plot.Plots import plot_cross_validation
from spio.ImageLoading import load_grayscale_image_dataset
from spio.Models import load_object, save_object, exists_checkpoint

CHECKPOINT_FILE = "adaboost.svd"
CROSS_VALIDATION = False

dataset, labels = load_grayscale_image_dataset()

labels = labels_to_int(labels)
dataset = normalize_dataset(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)
if CROSS_VALIDATION:
    adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=0x6),
                                             algorithm="SAMME",
                                             n_estimators=0x64,
                                             learning_rate=0.5,
                                             random_state=RANDOM_STATE)

    scores = cross_val_score(adaboost_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "AdaBoost  Score GS")
else:

    if exists_checkpoint(CHECKPOINT_FILE):
        adaboost_classifier = load_object(CHECKPOINT_FILE)
    else:
        param_grid = [{"learning_rate": [0.5, 1.0, 1.5], 'algorithm': ["SAMME"]}]

        adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=0x6),
                                                 algorithm="SAMME",
                                                 n_estimators=0x64,
                                                 learning_rate=0.5,
                                                 random_state=RANDOM_STATE)

        grid_search_cv = GridSearchCV(adaboost_classifier,
                                      param_grid,
                                      n_jobs=-0x1,
                                      verbose=0x1)

        grid_search_cv.fit(X_train, y_train)

        """
            grid_search_cv.best_params_ -> {'algorithm': 'SAMME', 'learning_rate': 0.5}
        """
        adaboost_classifier = grid_search_cv.best_estimator_
        save_object(adaboost_classifier, CHECKPOINT_FILE)
        y_pred = grid_search_cv.predict(X_test)

    disp = plot_confusion_matrix(adaboost_classifier,
                                 X_test,
                                 y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("AdaBoost Confusion Matrix GS")
    plt.show()
