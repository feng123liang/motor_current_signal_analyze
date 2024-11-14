from prepossess import *
import pandas as pd
import os
import glob
import numpy as np
import time
import socket
import threading
import ctypes

class RealtimeReader():

    def __init__(self,pumpid,one_step,low_factor,model_ref,mqtt_ref,device,udp_info):
        self.device = device
        # self.cv = threading.Condition()
        # self.groups = [[],[]]
        # self.index = 1
        self.lock = threading.Lock()
        self.pumpid = pumpid
        self.one_step = one_step
        self.low_factor = low_factor
        self._preprocess = Preprocess(model_ref)
        self.mqtt = mqtt_ref
        self.sample_indices = np.arange(0, one_step, low_factor) # 6400/16 = 50*25=1250

        # self.datas = [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
        self.datas = (ctypes.c_float * (6 * one_step*4))()
        # self.datas2 = 
        self.transfer = None
        self.index = 0
        self.count = 0
        self.flag = 0
        # udp_addr = udp_info.ad
        udp_addr = (udp_info['address'], udp_info['port'])
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定端口
        self.udp_socket.bind(udp_addr)
        # self.init_thread()
        

    def __del__(self):
        # 释放内存
        del self.data        

    def simple_update(self):
        while True:
            recv_data,addr = self.udp_socket.recvfrom(8192)  # 1024表示本次接收的最大字节数
            
    #        print(recv_data,addr)
            if(recv_data != None):
                # recv_data = decode_uint16_array(recv_data)
                leng =  len(recv_data)
                with self.lock:
                    # if(self.flag):
                    #     pointer = self.datas
                    for i in range(0, leng, 2):  
                        if i + 1 < leng:  
                            value = int.from_bytes(recv_data[i:i+2], byteorder='little', signed=True)  
                            self.datas[self.flag][self.index].append(value)
                            self.index = (self.index + 1)%6

    def send_data(self):
        # self._preprocess.get_batch(tem)
        # output = self._preprocess.fetch_data()
        # pre_lab = torch.argmax(output,dim = 1)
        flag = None
        while True:
            with self.lock:
                flag = self.flag
                self.flag = 1 - self.flag
                print(f"flag change into{self.flag} and current flag is {flag}")

            start_time = time.time()
            # if(flag == 0):
            self.transfer = np.array(self.datas[flag].copy()  )#,dtype=float)
            self.datas[flag] = [[],[],[],[],[],[]]
            print(f"the transfer shape is {self.transfer.shape}")

            try:
                self.one_step = self.transfer.shape[1]
                if(self.one_step > self.low_factor):
                    self.transfer = self.transfer * 2 * (3.3 / 4096)    
                    self.sample_indices = np.arange(0, self.one_step, self.low_factor)
            # else:
                # self.transfer = np.array(self.datas2.copy())
            # print(self.index,len(acquired_data))
            # self.groups[self.index] = []
            # self.index ^= 1
            # pre_lab = torch.randint(0,2)
                    self.mqtt.my_publish(self.pumpid,None,
                            0,self.transfer[:,self.sample_indices].tolist(),None,self.sample_indices)
            except:
                pass

            end_time = time.time()

            time.sleep(max(1 - end_time + start_time,0.001))
        
    def start_thread(self):

        update_thread = threading.Thread(target=self.simple_update)
        process_thread = threading.Thread(target=self.send_data)

        update_thread.start()
        time.sleep(1)
        process_thread.start()
