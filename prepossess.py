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

import numpy as np
import torch
import scipy.signal as signal

class Preprocess:
    def __init__(self, model_ref, model_device='cpu', sampling_rate=20000, stop_rate=3000, order=2):
        self.device = model_device
        self.sampling_rate = sampling_rate
        self.stop_rate = stop_rate
        self.order = order
        self.model = model_ref
    
    def band_stop_filter(self, data, fs, f_range, order):
        nyquist = 0.5 * fs
        low = f_range[0] / nyquist
        high = f_range[1] / nyquist
        b, a = signal.butter(order, [low, high], btype='bandstop', analog=False)
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data
    
    def get_filtered_data(self, data):
        time_domain_signal = data.reshape(-1)
        data_length = len(time_domain_signal)
        frequency_domain_signal = np.fft.fft(time_domain_signal, data_length)
        frequencies = np.fft.fftfreq(data_length, d=1/self.sampling_rate)
        frequency_domain_signal = np.abs(frequency_domain_signal) / data_length
        frequency_domain_signal = 20 * np.log10(frequency_domain_signal)
        
        index = np.argmax(frequency_domain_signal)
        greatest_freq = max(2, frequencies[index])
        
        filtered_data = self.band_stop_filter(time_domain_signal, self.sampling_rate, [greatest_freq - 2, greatest_freq + 2], order=2)
        
        b, a = signal.butter(self.order, self.stop_rate/(self.sampling_rate/2), 'lowpass')
        filtered_data = signal.filtfilt(b, a, filtered_data)
        
        return filtered_data

    def process_data(self, datas):
        processed_data = self.get_filtered_data(datas)
        
        for i in range(0, len(processed_data), self.one_group):
            tem = processed_data[i:i+6400]
            tem = (tem - min(tem)) / (max(tem) - min(tem)) * 255
            model_input = torch.Tensor(tem).to(self.device)
            self.model.to(self.device)
            tem_result = self.model(model_input)
            tem_result = tem_result.to("cpu")
            self.outputData.put(tem_result)
            self.data.put(processed_data[i:i+20000])

    def fetch_data(self):
        if self.outputData.empty():
            return None
        return self.outputData.get()
        
            


