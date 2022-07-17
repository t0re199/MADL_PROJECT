import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM

from image.ImagePreProcessing import labels_to_anomaly_detection, normalize_dataset
from plot.Plots import plot_roc_curve
from spio.ImageLoading import load_image_dataset
from spio.Models import load_object, save_object, exists_checkpoint

CHECKPOINT_FILE = "image_anomaly_svm.svd"
CROSS_VALIDATION = False

dataset, labels = load_image_dataset(flatten=True)

labels = labels_to_anomaly_detection(labels)
dataset = normalize_dataset(dataset)

regular_samples = dataset[np.argwhere(labels == 0x1)]
regular_samples = regular_samples.reshape(regular_samples.shape[0], -1)


outlier_probability = regular_samples.size / float(dataset.size)

if exists_checkpoint(CHECKPOINT_FILE):
    support_vector_classifier = load_object(CHECKPOINT_FILE)
else:
    param_grid = [
                    {"gamma": [0.001, 0.0001, 0.00001, 0.000004], "kernel": ["rbf", "poly"]}
                ]

    support_vector_classifier = OneClassSVM(nu=outlier_probability,
                                            gamma=0.0001,
                                            kernel="rbf")


    support_vector_classifier.fit(regular_samples)


    save_object(support_vector_classifier, CHECKPOINT_FILE)
    scores = support_vector_classifier.score_samples(dataset)

    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, scores)
    auc_score = auc(false_positive_rate, true_positive_rate)

    plot_roc_curve(true_positive_rate, false_positive_rate, auc_score)
