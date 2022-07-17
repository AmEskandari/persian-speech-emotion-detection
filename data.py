import os
import librosa 

import numpy as np

from tqdm import tqdm
from utils import addAWGN
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """ Explanation
        
        data, .......
        
        
         """
    def __init__(self,
                data_dir,
                waves_list,
                label_list,
                num_mel_filter,
                max_len = 3,
                offset = 0.5,
                sample_rate = 44100):
        """ exp """

        self._data_dir = data_dir 
        self._waves_list = waves_list
        self._label_list = label_list
        self._num_mel_filter = num_mel_filter 
        self._sr = sample_rate
        self._max_len = max_len
        self._offset = offset
        self._signals = []

    def generate_features(self):
        """
        Exp
        """
        
        if os.path.isfile(os.path.join(self._data_dir,'featurs_data.txt')):
            
            print('Features are ready. Loading done...')
            waves_features = np.loadtxt(os.path.join(self._data_dir,'featurs_data.txt'),
                                        delimiter=',')
            labels = np.loadtxt(os.path.join(self._data_dir,'label_data.txt'),
                                        delimiter=',')

        else:
            
            self._load_waves()
            self._augment_sinals()

            mfcc_list = []
            for idx in tqdm(range(self._signals.shape[0]),desc='Generating Features...'):
                mel_f = np.mean(librosa.feature.mfcc(y = self._signals[idx], 
                                            n_mfcc=self._num_mel_filter).T[:,:12],
                                            axis = 1).reshape(1,-1)
                mfcc_list.append(mel_f)

            waves_features = np.stack(mfcc_list,axis=0).reshape(-1,259)     
            labels = self._label_list

            np.savetxt(os.path.join(self._data_dir,'featurs_data.txt'),
                                        waves_features, delimiter=',')

            np.savetxt(os.path.join(self._data_dir,'label_data.txt'),
                                        labels, delimiter=',')                                        


        lb = LabelEncoder()
        labels = np_utils.to_categorical(lb.fit_transform(labels))
        
        del self._signals

        return waves_features, labels

    def splite_data(self,x_data,y_data, valid_ratio, test_ratio):
        """   
        
        
        """
        N = x_data.shape[0]
        train_ratio = valid_ratio+test_ratio

        train_x = x_data[:int(N * (1-train_ratio))]
        train_y = y_data[:int(N * (1-train_ratio))]

        valid_x = x_data[int(N * (1-train_ratio)):int(N * ((1-train_ratio)+valid_ratio))]
        valid_y = y_data[int(N * (1-train_ratio)):int(N * ((1-train_ratio)+valid_ratio))]

        test_x = x_data[int(N * ((1-train_ratio)+valid_ratio)):]
        test_y = y_data[int(N * ((1-train_ratio)+valid_ratio)):]

        print(f'Number of Samples: Train: {train_x.shape[0]} | Valid: {valid_x.shape[0]} | Test: {test_x.shape[0]}')

        return train_x,valid_x,test_x, train_y,valid_y,test_y

    def _load_waves(self):
        """  
        Expa

        """
        for file_name in tqdm(self._waves_list, desc='Loading the waves...'):

            audio, _ = librosa.load(os.path.join(self._data_dir,file_name),
                                    duration=self._max_len,offset=self._offset,
                                    sr= self._sr )
            
            signal = np.zeros((int(self._sr *self._max_len,)))
            signal[:len(audio)] = audio
            self._signals.append(signal)

        self._signals = np.stack(self._signals,axis=0)


    def _augment_sinals(self):
        """
        
        """
        aug_signals = []
        aug_labels = []
        
        for i in tqdm(range(self._signals.shape[0]), desc='Augmenting signals...'):
            signal = self._signals[i,:]
            augmented_signals = addAWGN(signal)
            for j in range(augmented_signals.shape[0]):
                aug_labels.append(self._label_list[i].item())
                aug_signals.append(augmented_signals[j,:])
                
            
        aug_signals = np.stack(aug_signals,axis=0)
        self._signals = np.concatenate([self._signals,np.stack(aug_signals,axis=0)],axis=0)
        aug_labels = np.stack(aug_labels,axis=0)
        self._label_list = np.concatenate([self._label_list,aug_labels])
        del aug_signals, aug_labels
        
        print('')
        print(f'Number of signals after augmnetion: {self._signals.shape[0]}')

