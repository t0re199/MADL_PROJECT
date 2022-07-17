import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM

from plot.Plots import plot_roc_curve
from spio.Models import load_object, save_object, exists_checkpoint
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset, vectorize_dataset, labels_to_anomaly_detection

CHECKPOINT_FILE = "anomaly_text_svm.svd"
CROSS_VALIDATION = False

dataframe = load_custom_dataset()

dataset = preprocess_dataset(np.copy(dataframe.Text.values))
labels = labels_to_anomaly_detection(dataframe.Score.values)

dataset = vectorize_dataset(dataset).toarray()

regular_samples = dataset[np.argwhere(labels == 0x1)]
regular_samples = regular_samples.reshape(regular_samples.shape[0], -1)


outlier_probability = regular_samples.shape[0] / float(dataset.shape[0])


if exists_checkpoint(CHECKPOINT_FILE):
    support_vector_classifier = load_object(CHECKPOINT_FILE)
else:
    support_vector_classifier = OneClassSVM(nu=outlier_probability,
                                            gamma=0.0001,
                                            kernel="rbf")

    support_vector_classifier.fit(regular_samples)


    save_object(support_vector_classifier, CHECKPOINT_FILE)

scores = support_vector_classifier.score_samples(dataset)

false_positive_rate, true_positive_rate, threshold = roc_curve(labels, scores)
auc_score = auc(false_positive_rate, true_positive_rate)

plot_roc_curve(true_positive_rate, false_positive_rate, auc_score)
