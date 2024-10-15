## 水泵项目组部分代码

### 模型代码
> BF_datamake.ipynb \
> BF_model.ipynb \
> BFdataMaker.py

### 接受、处理、传输数据代码
> client.py # mqtt_client 类的封装\
> model_run_demo.py # 总程序运行入口\
> model.py # 模型类定义\
> prepossess.py # 数据预处理封装 \
> predict_dict.json # defined error type code \
> udp_receiver.py # receiver data from hardware part 

### debug 代码
> plot.py # examine the received data and plot it 

### What's next to do?
> optimize the model \
> add threading module to the program entrance for forming a task managers for further developemnt \
> 目前用离线数据进行对接，后面需要封装udp_receiver.py，使之对接总程序