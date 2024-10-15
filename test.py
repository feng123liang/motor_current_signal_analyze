#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# # 创建一个随时间变化的示例 NumPy 数组
# time_steps = 100
# data = np.random.rand(time_steps, 10)  # 100个时间步, 每个时间步有10个值

# def draws():
#     global data
#     # 创建画布和子图
#     fig, ax = plt.subplots()
#     line, = ax.plot(data[0])

#     # 更新函数，每帧更新数据
#     def update(frame):
#         line.set_ydata(data[frame])
#         return line,

#     # 创建动画
#     ani = FuncAnimation(fig, update, frames=range(time_steps), blit=True)

# # 展示动画
#     plt.show()

exitFlag = 0
 
class udpThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print( "Starting " + self.name)
        print_time(self.name, self.counter, 5)
        print ("Exiting " + self.name)
 
def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print( "%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

class plotThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print( "Starting " + self.name)
        # 创建画布和子图
        fig, ax = plt.subplots()
        line, = ax.plot(self.data[0])

        # 更新函数，每帧更新数据
        def update(self):
            if(len(self.data ) >= 1000):
                line.set_ydata(self.data[-1000:])
                return line,

        # 创建动画
        ani = FuncAnimation(fig, update, frames=range(time_steps), blit=True)

        plt.show()


# 创建新线程
thread1 = udpThread(1, "udp", 1)
thread2 = plotThread(2, "", 2)
 
# 开启线程
thread1.start()
thread2.start()
 
print ("Exiting Main Thread")