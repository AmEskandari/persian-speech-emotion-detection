import os
import librosa 
import torch

import numpy as np


class DataLoaderCostum:
    """ Explanation
        
        data, .......
        
        
         """
    def __init__(self, data_dir, instance_list, win_len, win_hop,
                num_mel_filter, max_len, sample_rate):
        """ exp """

        self.data_dir = data_dir 
        self.instance_list = instance_list 
        self.win_len = win_len 
        self.win_hop = win_hop 
        self.num_mel_filter = num_mel_filter 
        self.sr = sample_rate
        self.max_len = max_len

    def get_instance(self, num_instance):
        """ This methods ...... """

        data_batch = []

        if len(self.instance_list) > num_instance:

            for num in range(num_instance):
                
                sig, sr = librosa.load(os.path.join(self.data_dir,
                                                    self.instance_list.pop()),
                                        sr = self.sr)
         
                sig = self._zero_pad(sig)
                data_batch.append(self._gen_feature(sig))
                

        elif len(self.instance_list) != 0: 
            
            iter_num = len(self.instance_list)

            for num in range(iter_num):

                sig, sr = librosa.load(os.path.join(self.data_dir,
                                                    self.instance_list.pop()),
                                        sr = self.sr)
                sig = self._zero_pad(sig)
                data_batch.append(self._gen_feature(sig))
                
        else:
            raise ValueError('There is no instance')

        return torch.FloatTensor(data_batch)

    def _zero_pad(self, data):
        """ This methods zero pad the signals, at ....."""
        
        return np.concatenate(([np.zeros(self.max_len - data.size),data]),axis = 0)


    def _gen_feature(self, data):
        """This methods generates the stakced features  """
        
        mel_f = librosa.feature.melspectrogram(y = data, sr = self.sr
                                               ,n_mels = self.num_mel_filter,
                                                hop_length=self.win_hop, win_length = self.win_len)
        
        rms_f = librosa.feature.rms(y = data, frame_length = self.win_len,
                                    hop_length=self.win_hop) 
        
        zcr_f = librosa.feature.zero_crossing_rate(y = data, frame_length=self.win_len,
                                                   hop_length=self.win_hop)
        
        return np.concatenate([mel_f, rms_f, zcr_f], axis = 0).T.tolist()

