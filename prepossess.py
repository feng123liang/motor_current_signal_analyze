# %%
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle # shuffle can shffule two array with the same random seed to maintain the mapping relation
import torch.utils.data as Data
import scipy.io as sio

import scipy.signal as signal
from scipy.signal import firwin, lfilter 

from scipy.fftpack import fft, ifft

import queue
import model
import torch


# %%
class Preprocess():
    def __init__(self,model_ref=None,model_device = 'cpu'):
        self.device = model_device
        self.data = queue.Queue()
        self.outputData = queue.Queue()
        self.sampling_rate = 20000
        self.stop_rate = 3000
        self.order = 2
        self.group_num = 10
        self.one_group = 20000
        if(model_ref == None):
            self.model = model.Model()
        else:
            self.model = model_ref
    def band_stop_filter(self,data, fs, f_range, order):
        nyquist = 0.5 * fs
        low = f_range[0] / nyquist
        high = f_range[1] / nyquist
        b, a = signal.butter(order, [low, high], btype='bandstop',  analog=False)
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data
    def get_filtered_data(self,data,sampling_rate=64000, stop_rate = 25000, order = 2,):
        time_domain_signal = data.reshape(-1)
        data_length = len(time_domain_signal)  # 数据长度

        frequency_domain_signal = fft(time_domain_signal,data_length)
        frequencies = np.fft.fftfreq(data_length, d=1/sampling_rate,)
        frequency_domain_signal = np.abs(frequency_domain_signal)
        frequency_domain_signal = frequency_domain_signal/data_length
        frequency_domain_signal = 20 * np.log10(frequency_domain_signal)
        # plt.plot(frequency_domain_signal)
        index = np.argmax(frequency_domain_signal, axis = 0)
        print('the frequencies index: ',frequencies[index])
        greatest_fre = max(2,frequencies[index])
        print('the greatest frequency:', greatest_fre)
        filtered_data = self.band_stop_filter(time_domain_signal,sampling_rate,[greatest_fre - 2, greatest_fre + 2], order = 2)
        fs = sampling_rate  # 采样率
        fc = stop_rate  # 截止频率
        order = order  # 滤波器阶数 不知有无影响？
        b, a = signal.butter(order, fc/(fs/2), 'lowpass')
        cdata = signal.filtfilt(b,a,filtered_data)
        return cdata
    def process_data(self,datas):
        processed_data = self.get_filtered_data(datas,self.sampling_rate,self.stop_rate,self.order)

        # can be improved
        for i in range(0,len(processed_data),self.one_group):
            tem = processed_data[i:i+6400]
            tem = (tem - min(tem)) / (max(tem) - min(tem)) * 255
            model_input = torch.Tensor(tem)
            model_input.to(self.device)
            self.model.to(self.device)
            temresult = self.model(model_input.to(self.device))
            temresult.to("cpu")
            self.outputData.put(temresult)

            # maybe left for fft data later, no idea 
            # self.data.put(processed_data[i:i+20000])

    def fetch_data(self):
        if(self.outputData.empty() == True):
            return False
        return self.outputData.get()
        
            