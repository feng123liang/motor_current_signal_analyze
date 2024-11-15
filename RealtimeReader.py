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
        self.pumpid = pumpid

        # one seconds how many data points in one channel
        self.one_step = one_step
        self.low_factor = low_factor
        self._preprocess = Preprocess(model_ref)
        self.mqtt = mqtt_ref
        # 用于降采样的index数组
        self.sample_indices = None
        
        # 方案一，用两个保存点来存数据
        # self.datas = [[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
        # planB, use manual memory management to reserve enough space to avoid data conflict between multithreading
        # self.datas = (ctypes.c_float * (6 * one_step*4))()
        # planC, use ndarray to reserve enough space and the import thing is that it's two-dimension easy to operate
        self.indexLim = one_step << 2
        self.datas = np.zeros((6,self.indexLim))
        # data will be sent to server by mqtt
        self.transfer = None
        self.index = 0 # 记录到了哪一路
        self.count = 0 # 记录到现在对数组的第几位进行操作
        self.flag = 0

        udp_addr = (udp_info['address'], udp_info['port'])
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定端口
        self.udp_socket.bind(udp_addr)
       
    def simple_update(self):
        while True:
            recv_data, addr = self.udp_socket.recvfrom(8192)  # 本次接收的最大字节数
    #        print(recv_data,addr)
            if(recv_data != None):
                leng =  len(recv_data)
                for i in range(0, leng, 2):  
                    if i + 1 < leng:  
                        value = int.from_bytes(recv_data[i:i+2], byteorder='little', signed=True)  
                        self.datas[self.index,self.count] = value
                        self.index += 1
                        if(self.index == 6):
                            self.index = 0
                            self.count += 1
                            if(self.count == self.indexLim):
                                self.count = 0

    def send_data(self):
        # self._preprocess.get_batch(tem)
        # output = self._preprocess.fetch_data()
        # pre_lab = torch.argmax(output,dim = 1)
        oldIndex = 0
        newIndex = 0
        while True:

            start_time = time.time()
            newIndex = self.count
            print(f"index change into{newIndex} and old index is {oldIndex}")

            # we have loop the whole ndarray
            if(newIndex == oldIndex):
                print("it maybe some error that the index didn't move or move too fast")
                pass
            elif(newIndex < oldIndex):
                self.transfer = np.hstack((self.datas[:,oldIndex:], self.datas[:,:newIndex])) #,dtype=float)
            else:
                self.transfer = self.datas[:,oldIndex:newIndex]#,dtype=float)
            # self.datas[flag] = [[],[],[],[],[],[]]
            print(f"the transfer shape is {self.transfer.shape}")

            self.one_step = self.transfer.shape[1]
            if(self.one_step > self.low_factor):

                # 要对电流进行映射
                self.transfer = self.transfer #* 2 * (3.3 / 4096)  
                for i in range(6):
                    upperBound = max(self.transfer[i,:])  
                    lowerBound = min(self.transfer[i,:])
                    midBound = (upperBound + lowerBound)/2
                    # print(f"index{i}, upperBound: {upperBound}, lowerbound: {lowerBound}, midbound: {midBound}")
                    self.transfer[i,:] = (self.transfer[i,:] - midBound)/2048*50
                    # print(f"after process max: {max(self.transfer[i,:])}, min: {min(self.transfer[i,:])}")

                self.sample_indices = np.arange(0, self.one_step, self.low_factor)

            self.mqtt.my_publish(self.pumpid,None,
                    0,self.transfer[:,self.sample_indices].tolist(),None,self.sample_indices)
         
            end_time = time.time()
            oldIndex = newIndex
            time.sleep(max(1 - end_time + start_time,0.001))
        
    def start_thread(self):

        update_thread = threading.Thread(target=self.simple_update)
        process_thread = threading.Thread(target=self.send_data)

        update_thread.start()
        time.sleep(1)
        process_thread.start()
