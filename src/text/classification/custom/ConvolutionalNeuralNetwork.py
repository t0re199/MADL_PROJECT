import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping

from Constants import RANDOM_STATE, EPOCHES, MAX_LEN
from plot.Plots import plot_history
from spio.Models import *
from spio.TextLoading import load_custom_dataset
from text.TextPreProcessing import preprocess_dataset

HISTORY_FILE = "text_cnn.hsvd"
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, "text_cnn.svd")
CROSS_VALIDATION = False
SEED = 33

dataframe = load_custom_dataset()

dataset = preprocess_dataset(np.copy(dataframe.Text.values))
labels = np.copy(dataframe.Score.values)

tokenizer = Tokenizer(num_words=MAX_LEN)
tokenizer.fit_on_texts(dataset)

dataset = tokenizer.texts_to_sequences(dataset)
corpus_len = len(tokenizer.word_index) + 0x1

dataset = pad_sequences(dataset, padding="post", maxlen=MAX_LEN)


def create_model(corpus_len, max_len=MAX_LEN):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=corpus_len,
                               output_dim=100,
                               input_length=max_len))

    model.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6, seed=SEED))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5, seed=SEED))
    model.add(layers.Dense(3, activation='softmax'))
    return model


X_train, X_test, y_train, y_test = train_test_split(dataset,
                                                    labels,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

model = create_model(corpus_len)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

if exists_checkpoint(HISTORY_FILE):
    model.load_weights(CHECKPOINT_FILE)
    history_dict = load_object(HISTORY_FILE)
else:
    model_checkpoint_callback = EarlyStopping(restore_best_weights=True,
                                              monitor='val_accuracy',
                                              mode='max',
                                              patience=2)
    history_object = model.fit(X_train,
                               y_train,
                               epochs=EPOCHES,
                               callbacks=[model_checkpoint_callback],
                               validation_data=(X_test, y_test))

    history_dict = history_object.history
    save_object(history_object.history, HISTORY_FILE)

plot_history(history_dict, "Convolutional Neural Network Accuracy")
