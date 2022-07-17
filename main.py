from calendar import EPOCH
import os
import json
import keras

import pandas as pd 
import numpy as np

from data import DataLoader
from model import get_model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

##############################################
################Hyperparamtetrs###############
##############################################
NUM_MEL_FILTERS = 39
EFFECTIVE_MEL_FILTERS = 12
RANDOM_SEED = 99123138
DATA_DIR = os.path.join(os.getcwd(),'DATA/male')
VALID_RATIO = 0.2
TEST_RATIO = 0.1
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
MOMENTUM = 0.0
WEIGHT_DECAY = 0.0
NUMBER_OF_CLASSES = 4
##############################################



##############################################
######Data Loading and pre-Processing#########
##############################################

rng = np.random.RandomState(RANDOM_SEED) 
#Reading the selected subset of dataset 
subset_metadata = pd.read_csv(os.path.join(os.getcwd(),
                              'DATA/meta_data/M_SubDataset_info.csv')).sample(frac=1,
                                                                       random_state=rng)
waves_list = subset_metadata['Name']
label_list = np.argmax(np.array(subset_metadata.iloc[:,-4:]), axis = 1)

#Instanciating the DataLoader object for feature generation and reading waves
data_obj = DataLoader(DATA_DIR, waves_list, label_list, NUM_MEL_FILTERS,EFFECTIVE_MEL_FILTERS)
#Get MFCCs features vectors for each waves
x_data, y_data = data_obj.generate_features()
#Splitting the train, test and validation data 
x_train,x_valid,x_test, y_train,y_valid,y_test = data_obj.splite_data(x_data,y_data, 
                                                                      VALID_RATIO,TEST_RATIO)
#Change the shape of the data suitable for cnn architecture 
x_train_cnn = np.expand_dims(x_train, axis=2)
x_valid_cnn = np.expand_dims(x_train, axis=2)
x_test_cnn = np.expand_dims(x_test, axis=2)


##############################################
######Preparing the optimizer and model#######
##############################################

#Creating the optimizer object
opt = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM, 
                            decay=WEIGHT_DECAY, nesterov=False)
#Creating the model object 
model = get_model(x_data.shape[1],NUMBER_OF_CLASSES)

print('Model Summery: \n')
model.summary()

#Prepating the model for training
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics= 'accuracy')

#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)

#Saving checkpoints and weights of the model in training phase
mcp_save = ModelCheckpoint(os.path.join(os.getcwd(),'saved_models/male_4class.h5'), 
                            save_best_only=True,monitor='val_loss', mode='min')

#Training The model
cnnhistory = model.fit(x_train_cnn, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                     validation_data=(x_valid_cnn, y_valid), callbacks=[mcp_save])

#Saving the model's info
model_json = model.to_json()
with open(os.path.join(os.getcwd(),'saved_models/male_4class.json'), "w") as json_file:
    json_file.write(model_json)


score = model.evaluate(x_valid_cnn, y_valid, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))