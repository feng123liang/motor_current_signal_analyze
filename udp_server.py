import time
import socket
import numpy as np
import pandas as pd
def decode_uint16_array(payload):  
    uint16_array = []  
    for i in range(0, len(payload), 2):  
        if i + 1 < len(payload):  
            value = int.from_bytes(payload[i:i+2], byteorder='little', signed=True)  
            uint16_array.append(value)
    return uint16_array  


def main():
    # global tot
    # udp 通信地址，IP+端口号
    udp_addr = ('10.7.154.160', 1883)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口
    udp_socket.bind(udp_addr)
    count = 0
    current_time = time.time()
    # 等待接收对方发送的数据
    tot = np.array([0])
    while True:
        recv_data,addr = udp_socket.recvfrom(8192)  # 1024表示本次接收的最大字节数
        
#        print(recv_data,addr)
        if(recv_data != None):
            recv_data = decode_uint16_array(recv_data)
            tot = np.hstack((tot,np.array(recv_data)))
            
            # with open ('data.json','a') as f:
            #     json.dump(recv_data,f)
#            print(len(decode_uint16_array(recv_data)))
#            print("received: ",recv_data.decode('utf-8'))
            count += 1
            if count % 20 == 0:
                new_time = time.time() - current_time
                if(new_time > 4):
                    datawrite = pd.DataFrame(tot)
                    print(f"time spent: {new_time}")
                    datawrite.to_csv("test.csv",mode = 'w',header=None)
                    print(f"package receive count {count}")
                    print(f"data_shape {new_time.shape}")
                    break
            #     new_time = time.time()
            #     print(len(decode_uint16_array(recv_data))) # 
            #     print(new_time - current_time)
            #     current_time = new_time
            #     count = 0
 
if __name__ == '__main__':
    print("udp server ")
    main()
