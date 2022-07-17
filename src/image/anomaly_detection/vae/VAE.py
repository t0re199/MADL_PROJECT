import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import roc_curve, auc

from image.ImagePreProcessing import labels_to_int, normalize_dataset, squarefy_rgb
from image.anomaly_detection.vae.vae_utils import compute_anomaly_scores, flip_labels
from plot.Plots import plot_roc_curve
from spio.ImageLoading import load_image_dataset
from spio.Models import exists_checkpoint

CHECKPOINT_FILE = "vae.svd"


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.0055 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=None, mask=None):
        pass


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_model(image_shape=(3, 80, 80), latent_space_dim=16):
    encoder_inputs = keras.Input(shape=image_shape)

    x = layers.Conv2D(128, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(encoder_inputs)
    x = layers.BatchNormalization()(x) #40x40 40x30

    x = layers.Conv2D(256, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #20x20

    x = layers.Conv2D(512, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #10x10 10x7

    x = layers.Conv2D(512, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #5x5 5x3

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    z_mean = layers.Dense(latent_space_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_space_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_space_dim,))

    x = layers.Dense(5 * 5 * 512, activation="relu")(latent_inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((512, 5, 5))(x) #5x5

    x = layers.Conv2DTranspose(512, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #10x10

    x = layers.Conv2DTranspose(256, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #20x20

    x = layers.Conv2DTranspose(128, 4, strides=2, activation="relu", padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization()(x) #40x40


    decoder_outputs = layers.Conv2DTranspose(3, 4, strides=2, activation="sigmoid", padding="same", data_format="channels_first")(x) #80x80
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return VAE(encoder, decoder)


dataset, labels = load_image_dataset()

dataset = squarefy_rgb(dataset)
dataset = normalize_dataset(dataset)

labels = labels_to_int(labels)


regular_samples = dataset[np.argwhere(labels == 1)]

regular_labels = labels[labels == 1]

regular_samples = regular_samples.reshape(regular_samples.shape[0], 3, 80, 80)

model = build_model()

if exists_checkpoint(CHECKPOINT_FILE):
    model.load_weights(CHECKPOINT_FILE)
else:

    rmsprop = tf.keras.optimizers.RMSprop(
        learning_rate=0.0001,
        rho=0.9,
        momentum=0.9,
        epsilon=1e-07,
    )
    model = build_model()
    model.compile(optimizer=rmsprop)
    model.fit(x=regular_samples,
              y=None,
              shuffle=True,
              epochs=350,
              batch_size=256)


dataset = dataset.reshape(dataset.shape[0], 3, 80, 80)

scores, values = compute_anomaly_scores(model, dataset, L=3)

labels = flip_labels(labels)

false_positive_rate, true_positive_rate, threshold = roc_curve(labels, values)
auc_score = auc(false_positive_rate, true_positive_rate)

plot_roc_curve(true_positive_rate, false_positive_rate, auc_score)
