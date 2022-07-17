import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from Constants import RANDOM_STATE, COLOR_MAP, FOLDS
from plot.Plots import plot_cross_validation
from spio.Models import load_object, exists_checkpoint
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset, vectorize_dataset

CHECKPOINT_FILE = "text_knn.svd"
CROSS_VALIDATION = False

dataframe = load_custom_dataset()

dataset = preprocess_dataset(np.copy(dataframe.Text.values))
labels = dataframe.Score.values

dataset = vectorize_dataset(dataset).toarray()

X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

if CROSS_VALIDATION:
    knn_classifier = KNeighborsClassifier(leaf_size=0xf,
                                          n_neighbors=0x1e,
                                          weights="distance")

    scores = cross_val_score(knn_classifier,
                             dataset,
                             labels,
                             scoring="accuracy",
                             cv=FOLDS,
                             n_jobs=-0x1)

    plot_cross_validation(scores, "K-Nearest Neighbour Score")

else:
    if exists_checkpoint(CHECKPOINT_FILE):
        knn_classifier = load_object(CHECKPOINT_FILE)
    else:
        param_grid = [{"n_neighbors": [10, 15, 30], "weights": ["uniform", "distance"]}]

        knn_classifier = KNeighborsClassifier(leaf_size=0xf,
                                              n_neighbors=0x1e,
                                              weights="distance")

        grid_search_cv = GridSearchCV(knn_classifier,
                                      param_grid,
                                      n_jobs=-0x1,
                                      verbose=0x1)

        #grid_search_cv.fit(X_train, y_train)

        knn_classifier.fit(X_train, y_train)
        """
            grid_search_cv.best_params_ -> {'n_neighbors': 30, 'weights': 'distance'}
        """
        knn_classifier = grid_search_cv.best_estimator_
        #save_object(knn_classifier, CHECKPOINT_FILE)
        y_pred = knn_classifier.predict(X_test)

    disp = plot_confusion_matrix(knn_classifier, X_test, y_test,
                                 display_labels=np.unique(y_test),
                                 cmap=COLOR_MAP,
                                 normalize="true")

    disp.ax_.set_title("K-Nearest Neighbour Confusion Matrix")
    plt.show()
