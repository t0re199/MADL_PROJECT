import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from Constants import MAX_LEN
from plot.Plots import plot_roc_curve
from spio.Models import *
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset

HISTORY_FILE = "text_autoencoder.hsvd"
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, "text_autoencoder.svd")
CROSS_VALIDATION = False


def create_model(max_len=MAX_LEN):
    sequence_input = keras.Input(shape=(max_len,))
    embedded_sequences = layers.Embedding(input_dim=corpus_len,
                                          output_dim=100,
                                          input_length=max_len)(sequence_input)

    encoder = layers.Conv1D(filters=8, kernel_size=5, strides=2, activation='relu', padding='same')(
        embedded_sequences)  # 1000
    encoder = layers.Conv1D(filters=16, kernel_size=3, strides=2, activation='relu', padding='same')(encoder)  # 500
    encoder = layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(encoder)  # 250

    decoder = layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same')(encoder)
    decoder = layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, activation='relu', padding='same')(decoder)
    decoder = layers.Conv1DTranspose(filters=1, kernel_size=5, strides=2, activation='relu', padding='same')(decoder)
    decoder = layers.Flatten()(decoder)
    decoder = layers.Dense(max_len)(decoder)
    return keras.Model(sequence_input, decoder)


MAX_LEN = 2048

dataframe = load_custom_dataset()

dataset = preprocess_dataset(np.copy(dataframe.Text.values))
labels = np.copy(dataframe.Score.values)

tokenizer = Tokenizer(num_words=MAX_LEN)
tokenizer.fit_on_texts(dataset)

dataset = tokenizer.texts_to_sequences(dataset)
corpus_len = len(tokenizer.word_index) + 0x1

dataset = pad_sequences(dataset, padding="post", maxlen=MAX_LEN)

model = create_model(corpus_len)

regular_samples = dataset[np.argwhere(labels == 0x1)]
regular_samples = regular_samples.reshape(regular_samples.shape[0], -1)

model.compile(loss="mse",
              optimizer="adam",
              metrics=["accuracy"])

history_object = model.fit(regular_samples,
                           regular_samples,
                           epochs=700)

history_dict = history_object.history

scores = np.zeros(labels.shape[0], np.float64)
for i in range(labels.shape[0]):
    scores[i] = model.evaluate(dataset[[i]], dataset[[i]], verbose=0)

false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
auc_score = auc(false_positive_rate, true_positive_rate)

plot_roc_curve(true_positive_rate, false_positive_rate, auc_score)
