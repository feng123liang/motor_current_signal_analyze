# from DataGenerator import *
from prepossess import *
import pandas as pd
import os
import glob
import numpy as np
import time

class Simulater():

    def __init__(self,pump_id,target_folder,one_step,low_factor,model_ref,mqtt_ref,device):
        self.device = device
        self.data = 0
        # self.lock = threading.Lock()
        # self.cv = threading.Condition()
        # self.groups = [[],[]] 
        # self.index = 1
        self.pump_id = pump_id
        self.index = 0
        self.one_step = one_step
        self.low_factor = low_factor
        self._preprocess = Preprocess(model_ref,device)
        self.mqtt = mqtt_ref
        self.sample_indices = np.arange(0, one_step, low_factor) # 6400/16 = 50*25=1250

        self.datas = [[],[],[],[],[],[]]
        # 获取目标文件夹下所有后缀为 .csv 的文件路径
        csv_files = glob.glob(os.path.join(target_folder, "*.csv"))

        # 按文件路径排序
        csv_files.sort()

        # 获取前6个最小路径
        first_six_paths = csv_files[:6]

        print(first_six_paths)

        for i in range(6):
            self.datas[i] = pd.read_csv(first_six_paths[i]).values[:,1]
        self.datas = np.array(self.datas)
        print(self.datas.shape)
        self.highindex = self.datas.shape[1]
        # self.init_thread()

    def simple_update(self):
        if(self.index + self.one_step >= self.highindex):
            sub = self.index + self.one_step - self.highindex
            # tem = self.datas[0,self.index:].copy() 
            tem = np.hstack((self.datas[0,self.index:],self.datas[0,:sub]))
            transfer = np.hstack((self.datas[:,self.index:],self.datas[:,:sub]))
            # print(transfer.shape)
            self.index = sub
        else:
            tem = self.datas[0,self.index:self.index+self.one_step]
            transfer = self.datas[:,self.index:self.index+self.one_step]
            self.index += self.one_step

        self._preprocess.process_data(tem)
        output = self._preprocess.fetch_data()
        pre_lab = torch.argmax(output,dim = 1)
        self.mqtt.my_publish(self.pump_id,None,
                     pre_lab.item(),transfer[:,self.sample_indices].tolist(),None,self.sample_indices)
    
    def simple_timer(self):
        while True:
            start_time = time.time()
            self.simple_update()
            end_time = time.time()
            time.sleep(max(1 - end_time + start_time,0.001))
    # def update_data(self):
    #     while True:
    #         with self.cv:
    #             self.cv.wait()

    #             self.cv.notify()
    #             self.cv.release()
    #         while True:
    #             with self.lock:
    #                 acquired_data = self.groups[self.index].copy()
    #                 print(self.index,len(acquired_data))
    #                 self.groups[self.index] = []
    #                 self.index ^= 1

    # def publish_data(self):
    #     while True:
    #         with self.cv:
    #             self.cv.wait()
    #             acquired_data = self.groups[self.index].copy()
    #             # print(self.index,len(acquired_data))
    #             self.groups[self.index] = []
    #             self.index ^= 1
    #             self.cv.notify()
    #             self.cv.release()
    #             time.sleep(1)
    #         with self.lock:
                

            

    