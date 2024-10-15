import time
import random
import json
from paho.mqtt import client as mqtt_client
import numpy as np
class MQTTClient:
    def __init__(self, broker='10.5.153.251', port=1883, topic="default", id = 0):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = f'pump-mqtt-{id}'
        self.client = self.connect_mqtt()
        self.status = False
        self.current_time = 0
        self.sampling_rate = 20000
        self.sub_time_group = np.arange(1,0,-1/self.sampling_rate)
        self.time_group = None
    def connect_mqtt(self):
        def on_connect(self,client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        try:
            client = mqtt_client.Client(self.client_id)
        except:
            client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, self.client_id)

        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    
    def get_status(self):
        current_time = int(time.time())
        if(current_time > self.current_time):
            self.current_time = current_time
            return True
        return False
    def my_publish(self,pumpid,time_data,predict_status,time_domain_data,fft_domian_data,sample_indice):
        tem = ((np.full(20000,time.time()) - self.sub_time_group))
        tem = tem[sample_indice]
        print(tem.shape)
        tem = tem.tolist()
        msg = {'pump_id': f"pump_{pumpid}",
               'pump_station_id':f"station_{pumpid}",
               'pump_place':'default',
                   'time': tem,
                   'predict_status': predict_status,
                   'time_domain_data': {'i1':time_domain_data[0],
                                        'i2':time_domain_data[1],
                                        'i3':time_domain_data[2],
                                        'v1':time_domain_data[3],
                                        'v2':time_domain_data[4],
                                        'v3':time_domain_data[5],
                                        },
                   'fft_domain_data': fft_domian_data
                }
        # print()
        print(msg)
        msg_str = json.dumps(msg)
        # print(msg_str)
        # result = self.client.publish(f'topic/{pumpid}', msg_str)
        result = self.client.publish('pump/realtime_power_events', msg_str)
        # status = result[0]
        # print(result)
        status = 0 # ?
        if status == 0:
            print(f"Send success to topic `topic/{pumpid}`")
        else:
            print(f"Failed to send message to topic {self.topic}")
        # msg_count += 1

    def run(self):
        self.client.loop_start()
        # self.publish()

# Example of how to use the MQTTClient class
if __name__ == '__main__':
    mqtt_client = MQTTClient()
    mqtt_client.run()