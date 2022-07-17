from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D



def get_model(input_dim_size,number_of_class):
    """
    
    
    """

    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(input_dim_size,1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())


    model.add(Dense(number_of_class))
    model.add(Activation('softmax'))
    
    return model