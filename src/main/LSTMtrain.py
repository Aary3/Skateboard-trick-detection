from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from LSTMmodel import LSTMmodel
import time
import os.path
import sys

class LSTMtrain:
    def __init__(self, model, batch_size=32, epochs=100, validation_split=0.2):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.callbacks = []
        self.add_callback(self.callbacks)
        self.history = None

    def add_callback(self, callback):
        checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', self.model + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

        tb = TensorBoard(log_dir=os.path.join('data', 'logs', self.model))

        early_stopper = EarlyStopping(patience=5)

        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join('data', 'logs', self.model + '-' + 'training-' + \
            str(timestamp) + '.log'))   
        
        self.callbacks.append(checkpointer)
        self.callbacks.append(tb)
        self.callbacks.append(early_stopper)
        self.callbacks.append(csv_logger)

    def train(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_split=self.validation_split,
                                      callbacks=self.callbacks)