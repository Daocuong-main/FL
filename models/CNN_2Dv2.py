from tensorflow import keras
from keras.layers import TimeDistributed, Conv1D, MaxPool1D, Flatten, LSTM, Dense, AveragePooling1D, Dropout, Conv2D, MaxPool2D, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2


def create_keras_model(NUM_FEATURE, NUM_CLASSES):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    return model
