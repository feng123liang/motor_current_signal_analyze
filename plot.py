import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file',type = str, default="test.csv")
parser.add_argument('-c','--column',type=int,default=3)
parser.add_argument('-sc','--start_column',type=int,default=1)
args = parser.parse_args()

data = pd.read_csv(args.file,header = None).values
print(data.shape)
data = data[:,-1]
start_index = args.start_column
step = args.column
for i in range(start_index,start_index+step):
    cdata = data[i:len(data):step]
    # cdata = cdata[1000:2000]
    cdata = cdata * (3.3 / 4096)
    plt.figure(i,figsize=(20,4))
    plt.plot(cdata)
plt.show()

# 后面想要多线程
# import threading
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

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

# def sub():

#     for i in range(10):
#         time.sleep(1)
#         print(i)

# thread1 = threading.Thread(draws)
# thread2 = threading.Thread(sub)
# thread1.start()
# thread2.start()