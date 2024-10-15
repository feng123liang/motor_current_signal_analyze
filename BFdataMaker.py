# %%
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle # shuffle can shffule two array with the same random seed to maintain the mapping relation
import torch.utils.data as Data
import scipy.io as sio

import numpy as np
import scipy.signal as signal
from scipy.signal import firwin, lfilter 

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# %%
def get_8080_one_line_current_data_with_label_from_keyword(keyword,label):
    
    filelist = []
    for root,_,fileL in os.walk("raw_data"):
        for eachfile in fileL:
            if eachfile.endswith(".csv") and (eachfile.startswith(str(label)) or eachfile.count(keyword)):
                filelist.append(os.path.join(root,eachfile))
    rdata = np.array([0 for i in range(6400)])
    rdata = np.hstack((-1,rdata))
    output_file = "processed_data\\" + keyword + "_pixelvalue.csv"
    for eachfile in filelist:
        
        print(f"getting data from file {eachfile}")
        df = pd.read_csv(eachfile)
        df = df.values[:,1:]
        for cid in range(1,df.shape[1]):
            data = df[:,cid]
            length = data.shape[0]
            for startid in range(0,length-6400,6400):
                tem = np.hstack((label,data[startid:startid+6400])).reshape(1,6401)
                pd.DataFrame(tem).to_csv(output_file, mode="a", header=False, index=False)
           

# %%
def load_mat_to_two_current_filtered(matpath):
    # 格式化读取.mat文件，返回两相电流的数据
    # if not matname.endswith('.mat'):
    #     matnamewithmat = matname[:] + '.mat'
    # else:
    #     matnamewithmat = matname[:]
    #     matname = matname[:-4]
    if(matpath.find('\\')):
        matname = os.path.basename(matpath)[:-4]
    print(matname,matpath)
    # 加载 .mat 文件
    data = sio.loadmat(matpath)
    data = data[matname]
    data = data['Y']
    data = data[0][0][0]
    
    cdata1 = data[1][2].reshape(-1)
    cdata2 = data[2][2].reshape(-1)
    
    fs = 64000  # 采样率
    fc = 25000  # 截止频率
    order = 2  # 滤波器阶数 不知有无影响？
    b, a = signal.butter(order, fc/(fs/2), 'lowpass')

    cdata1 = signal.filtfilt(b,a,cdata1)
    cdata2 = signal.filtfilt(b,a,cdata2)
    cdata = np.vstack((cdata1,cdata2))
    return cdata


# %%
def get_fft_from_tdsignal(time_domain_signal,sampling_rate=64000):

    # 设置参数
    data_length = len(time_domain_signal)  # 数据长度    
    # window_function = np.hanning(data_length)  # 窗口函数

    # 进行 FFT
    frequency_domain_signal = fft(time_domain_signal,data_length)
   
    index = np.argmax(frequency_domain_signal, axis = 0)
   
    # 计算频率
    frequencies = np.fft.fftfreq(data_length, d=1/sampling_rate)
    
    
    # print(frequencies.shape,frequency_domain_signal.shape)
    # 绘制频谱图
    frequency_domain_signal = np.abs(frequency_domain_signal)
    frequency_domain_signal = frequency_domain_signal/data_length
    frequency_domain_signal = 20 * np.log10(frequency_domain_signal)

    index = np.argmax(frequency_domain_signal, axis = 0)
    
    # print(frequencies[index])

    # greatest_fre = max(2,frequencies[index])

    # filtered_data = band_stop_filter(time_domain_signal,fs,[greatest_fre - 2, greatest_fre + 2], order = 2)

    # 设置 x 轴范围
    return frequency_domain_signal[:2500]
    


# %%
def get_data_and_convert_to_pixels(keyword, label, step, td_length = 6400, fft_length = 64000, data_dir="raw_data", output_dir="processed_data"):
    """
    获取指定关键词和标签的数据，并转换为像素值。

    Args:
        keyword (str): 关键词。
        label (str): 标签。
        step (int): 步长。
        data_dir (str, optional): 数据目录。 Defaults to "raw_data".
        output_dir (str, optional): 输出目录。 Defaults to "processed_data".
    """

    output_file = os.path.join(output_dir, f"{keyword}_pixelvalue.csv")

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv") and (file.startswith(str(label)) or keyword in file):
                file_path = os.path.join(root, file)
                print(f"Getting data from file {file_path}")

                df = pd.read_csv(file_path)
                data = df.values[:, 1:]
                for cid in range(1, data.shape[1]):
                    current_data = data[:, cid]
                    length = current_data.shape[0]

                    for start_id in range(0, length - 6400, step):
                        data_segment = current_data[start_id:start_id + 6400]
                        normalized_data = (data_segment - min(data_segment)) / (max(data_segment) - min(data_segment)) * 255
                        pixel_data = np.hstack((label, normalized_data))
                        pixel_data = pixel_data.reshape(1, 6401)

                        pd.DataFrame(pixel_data).to_csv(output_file, mode="a", header=False, index=False)

            elif file.endswith(".mat") and (keyword in file):
                file_path = os.path.join(root, file)
                data = load_mat_to_two_current_filtered(file_path)
                for cid in range(data.shape[0]):
                    current_data = data[cid, :]
                    length = current_data.shape[0]

                    for end_id in range(fft_length, length, step): 
                        data_segment = current_data[end_id - step :end_id].copy()
                        fft_data = get_fft_from_tdsignal(current_data[end_id - fft_length :end_id])

                        normalized_data = (data_segment - min(data_segment)) / (max(data_segment) - min(data_segment)) * 255
                        pixel_data = np.hstack((label, normalized_data,fft_data))
                        pixel_data = pixel_data.reshape(1, -1)

                        temfile = output_file[:-4] + ".csv"
                        pd.DataFrame(pixel_data).to_csv(temfile, mode="a", header=False, index=False)
                    

# %%
def load_data_and_convert_to_grid(datadicts, ratio, data_dir="processed_data", length=6401):
    """
    加载数据并转换为网格。

    Args:
        datadicts (dict): 数据字典。
        ratio: float, train:test占比
        data_dir (str, optional): 数据目录。 Defaults to "processed_data".
        length: 一个 数据样本大小：包含时域频域以及标签
    Returns:
        tuple: 训练集和验证集。
    """
    # print('here')
    total_train_set = np.zeros(length)
    total_val_set = np.zeros(length)

    for keyword, file in datadicts.items():
        file_path = os.path.join(data_dir, f"{keyword}_pixelvalue.csv")
        print(f"Getting data with keyword {keyword}")

        df = pd.read_csv(file_path, header=None)
        data = df.values

        np.random.shuffle(data)
        sub_train_set, sub_val_set = data[:int(ratio * len(data))], data[int(ratio * len(data)):]

        total_train_set = np.vstack((total_train_set, sub_train_set.copy()))
        total_val_set = np.vstack((total_val_set, sub_val_set.copy()))

    
    total_train_set = total_train_set[1:]
    total_val_set = total_val_set[1:]
    # np.random.shuffle(total_set)
    # train_set, val_set = total_set[:int(ratio * len(total_set))], total_set[int(ratio * len(total_set)):]

    train_y = total_train_set[:, 0]
    train_x = total_train_set[:, 1:]
    val_y = total_val_set[:, 0]
    val_x = total_val_set[:, 1:]

    print(train_y.shape, train_x.shape)
    print(val_y.shape, val_x.shape)

    return train_x, train_y, val_x, val_y
# %%
def get_grid_current_data_with_label_from_keyword(keyword,label=0):
    
    filelist = []                                  
    for root,_,fileL in os.walk("raw_data"):
        for eachfile in fileL:
            if eachfile.endswith(".csv") and (eachfile.startswith(str(label)) or eachfile.count(keyword)):
                filelist.append(os.path.join(root,eachfile))
    # rdata = np.array([0 for i in range(6400)])
    # rdata = np.hstack((-1,rdata))
    # output_file = "processed_data\\" + keyword + "_pixelvalue.csv"
    for eachfile in filelist:
        
        print(f"getting data from file {eachfile}")
        df = pd.read_csv(eachfile)
        df = df.values[:,1:]
        return df

import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
 # Flatten confusion matrix for CSV
    
def evaluate_metrics(y_true, y_pred, csv_path, num_epoch, record = True):
    sns.set_theme(color_codes=True)

    # 计算各项评估指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'[Info] acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}')

    data = pd.DataFrame(np.array([acc, precision, recall, f1]).reshape(1, -1))
    # 混淆矩阵
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

    if(record == True):
        # Write metrics and confusion matrix data to CSV
        with open(csv_path, 'a') as file:
            file.write(f"___Epoch___{num_epoch}:\n")
            data.to_csv(file, header=False, index=False)
            pd.DataFrame(cf_matrix).to_csv(file, header=False, index=False)
        
    return acc


if __name__ == "__main__":
    keywordList = ["N15_M07_F04_"]
    keywordList = ["N09_M07_F10_" ,"N15_M07_F10_","N15_M01_F10_"]
    dicts = {"K001":0,"K002":0,'K003':0,"K004":0,"K005":0,
             "KA04":1,"KA15":1,"KA16":1,"KA22":1,"KA30":1,
             "KI04":2,"KI14":2,"KI16":2,"KI18":2,"KI21":2}
    # dicts = {"KI04":2,"KI14":2,"KI16":2,"KI18":2,"KI21":2}
    
    # for eachbt in dicts:
    #     for eachcondi in keywordList:
    #         print(f"\"{eachcondi + eachbt}\": {dicts[eachbt]}," ,end = '')

    folderName = "G_lp_td_fd_pixel"

    datadicts = {"N15_M01_F10_K001":0,"N15_M07_F10_K001": 0}
    for each in datadicts:
        get_data_and_convert_to_pixels(each,datadicts[each],6400,6400,128000,output_dir=folderName)
    # for eachbt in dicts:
    #     for eachcondi in keywordList:
    #         get_data_and_convert_to_pixels(eachcondi + eachbt,dicts[eachbt],6400,6400,128000,output_dir=folderName)

