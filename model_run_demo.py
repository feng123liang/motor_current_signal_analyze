

import torch
import time
from model import *
import numpy as np
import pandas as pd
import datetime
from client import *
import prepossess as pp
from Simulater import *
from RealtimeReader import *
import threading
# 格式化输出当前时间
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

model_name = 'test_4_200'
_model = Modelmp()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(device)
model_state_dict = torch.load('pth/' + model_name + '.pth')
_model.eval()
_model.load_state_dict(model_state_dict)


# 10.101.2.70 为目标计算机的地址
_mqtt = MQTTClient(broker="10.101.2.70")
_mqtt.run()

wholeTime = time.time()
# 到时候会写到专门的文件，从文件读入predict_dict
# predict_dict = {0:'HEALTHY',1:'BROKEN_ROTOR_BAR'}
one_step = 20000
group_num = 10
low_factor = 16 # 16
sample_indices = np.arange(0, one_step, 5000) # 6400/16 = 50*25=1250

udp_info = {'address':'10.42.0.1','port':1883}

_simulater = Simulater(1,"pump_100_BRB_H_filtered_cated",20000,33,_model,_mqtt,device)
_simulater2 = Simulater(2,"pump_100_BRB_H_filtered_cated",20000,33,_model,_mqtt,device)
_realTimeReader = RealtimeReader(3,8100,14,_model,_mqtt,device,udp_info)
# _simulater.simple_timer()

update_thread = threading.Thread(target=_simulater.simple_timer)
process_thread = threading.Thread(target=_simulater2.simple_timer)
realtime_thread = threading.Thread(target=_realTimeReader.start_thread)

update_thread.start()
process_thread.start()
realtime_thread.start()

update_thread.join()
process_thread.join()
realtime_thread.join()

wholeTime = time.time() - wholeTime
print("in total:", wholeTime)