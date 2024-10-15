import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("test.csv",header = None).values
print(data.shape)
data = data[:,1]

for i in range(1,4):
    cdata = data[i:len(data):3]
    # cdata = cdata[1000:2000]
    cdata = cdata * (3.3 / 4096)
    plt.figure(i,figsize=(20,4))
    plt.plot(cdata)
plt.show()
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