from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

class LSTMmodel:
    def __init__(self, input_shape, num_classes, sequence_length=2048):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-6, decay=1e-6), metrics=['accuracy'])           #categorical_crossentropy for more than 2 classes or binary_crossentropy for 2 classes, sparse_categorical_crossentropy for 2 classes with one-hot encoding  #learning_rate=1e-6, decay=1e-6
        self.sequence_length = sequence_length

    def build_model(self):
        model = Sequential()
        model.add(LSTM(1024, return_sequences=False, input_shape = self.input_shape, dropout=0.5))  #2048 lstm
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='sigmoid'))        #softmax, sigmoid
        return model
