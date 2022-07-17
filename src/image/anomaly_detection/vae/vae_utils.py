import numpy as np
import tensorflow as tf

DISABLED__ = -1.0


def compute_anomaly_scores(vae, anomalous_dataset, threshold=DISABLED__, L=5):
    labels = np.ones(anomalous_dataset.shape[0])
    scores = []
    bce = tf.keras.losses.BinaryCrossentropy()

    for i in range(anomalous_dataset.shape[0]):
        sample = anomalous_dataset[[i]]
        mean, variance, encoded = vae.encoder.predict(sample)

        variance = np.exp(variance)

        one = np.ones_like(mean)
        zero = np.zeros_like(mean)

        reconstruction_probability = 0.0
        for l in range(L):
            item = np.random.normal(zero, one)
            item = mean + variance * item
            decoded = vae.decoder(item)
            reconstruction_probability += bce(sample, decoded).numpy()

        reconstruction_probability /= L
        scores.append(reconstruction_probability)

        if threshold != DISABLED__:
            if reconstruction_probability < threshold:
                labels[i] = -1

    return scores, labels


def flip_labels(labels):
    labels_ = np.zeros_like(labels)
    labels_[np.argwhere(labels == 1)] = -1
    labels_[np.argwhere(labels == -1)] = 1
    return labels_
