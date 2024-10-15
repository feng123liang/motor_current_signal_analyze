

import torch
import time
from model import *
import numpy as np
import pandas as pd
import datetime
from client import *
import prepossess as pp
# 格式化输出当前时间
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    
groups = ["pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch1.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch2.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch3.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch4.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch5.csv",
          "pump_100_BRB_H_filtered_cated/Electric_Motor-2_100_time-broken rotor bar-ch6.csv"
          ]

datas = pd.read_csv(groups[0]).values[:,1]
for i in range(1,6):
    datas = np.vstack((datas,pd.read_csv(groups[i]).values[:,1]))

model_name = 'mp_brb__8'
model = Modelmp()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_state_dict = torch.load('pth/' + model_name + '.pth')
model.eval()
model.load_state_dict(model_state_dict)

_prepossess = pp.Prepossess(model)
# mqtt_ = MQTTClient(broker="10.108.4.125")
# 10.101.2.70 为目标计算机的地址
mqtt_ = MQTTClient(broker="10.101.2.70")
mqtt_.run()

index = 0
highindex = (datas.shape[1])
print(highindex)

wholeTime = time.time()
# 到时候会写到专门的文件，从文件读入predict_dict
predict_dict = {0:'HEALTHY',1:'BROKEN_ROTOR_BAR'}
one_step = 20000
group_num = 10
low_factor = 16 # 16
sample_indices = np.arange(0, one_step, 5000) # 6400/16 = 50*25=1250


while 1:
    since = time.time()
    

    if(index + one_step >= highindex):
        sub = index + one_step - highindex
        tem = datas[0,index:].copy() 
        tem = np.hstack((tem,datas[0,:sub].copy()))
        transfer = datas[:,index:]
        transfer = np.hstack((transfer,datas[:,:sub]))
        # print(transfer.shape)
        index = sub
    else:
        tem = datas[0,index:index+one_step].copy()
        transfer = datas[:,index:index+one_step]
        index += one_step
    # print(index, tem.shape)
    _prepossess.get_batch(tem)
    output = _prepossess.fetch_data()
    
    # print(output)
    pre_lab = torch.argmax(output,dim = 1)
    print(pre_lab)
    
    while(mqtt_.get_status()==False):
        # print('wait')
        time.sleep(0.05)
    
    mqtt_.my_publish(0,None,
                     predict_dict[pre_lab.item()],transfer[:,sample_indices].tolist(),None,sample_indices)
    print(time.time())
    time_use = time.time() - since
    # print(f"one detection and submission costing time: {time_use} seconds")
    # print("in total: ",  time.time() - wholeTime)
wholeTime = time.time() - wholeTime
print("in total:", wholeTime)