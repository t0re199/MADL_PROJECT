from threading import Thread

import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping

from Constants import EPOCHES


class KerasModelFitPerformer(Thread):
    def __init__(self, model, scores, fold, x_train, x_test, y_train, y_test, epoches=EPOCHES, early_stopping=False):
        Thread.__init__(self)
        self.model = model
        self.scores = scores
        self.fold = fold
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.epoches = epoches
        self.early_stopping = early_stopping


    def run(self):
        if self.early_stopping:
            early_stopping = EarlyStopping(restore_best_weights=True,
                                                      monitor='val_accuracy',
                                                      mode='max')

            history_object = self.model.fit(self.x_train,
                                            self.y_train,
                                            epochs=self.epoches,
                                            callbacks=[early_stopping],
                                            validation_data=(self.x_test, self.y_test))
        else:
            history_object = self.model.fit(self.x_train,
                                            self.y_train,
                                            epochs=self.epoches,
                                            validation_data=(self.x_test, self.y_test))

        self.scores[self.fold] = np.max(history_object.history["val_accuracy"])
