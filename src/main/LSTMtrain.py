from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from LSTMmodel import LSTMmodel
import time
import os.path
import os
import sys
import tensorflow as tf

class LSTMtrain:
    def __init__(self, model, modelName, batch_size=32, epochs=100, validation_split=0.2):
        self.model = model
        self.modelName = modelName
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.callbacks = []
        self.add_callback(self.callbacks)
        self.history = None

    def add_callback(self, callback):
        checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', self.modelName + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5.keras'),
        verbose=1,
        save_best_only=True)

        tb = TensorBoard(log_dir=os.path.join('data', 'logs', self.modelName))

        early_stopper = EarlyStopping(patience=5)

        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join('data', 'logs', self.modelName + '-' + 'training-' + \
            str(timestamp) + '.log'))   
        
        self.callbacks.append(checkpointer)
        self.callbacks.append(tb)
        self.callbacks.append(early_stopper)
        self.callbacks.append(csv_logger)

    def train(self, X_train, y_train):
        print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
        self.history = self.model.model.fit(X_train, y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_split=self.validation_split,
                                      callbacks=self.callbacks)