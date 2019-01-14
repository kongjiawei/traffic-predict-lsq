# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:53:47 2019

@author: 孔嘉伟
"""
import matplotlib.pyplot as plt
import os
import socket
import pandas as pd
import xlrd
import numpy as np
from keras.models import load_model
import json

#取平均
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

#初始化
dataset_node1=[]
dataset_node2=[]
dataset_node3=[]
dataset_node4=[]
dataset_node5=[]
dataset_node6=[]
dataset_traffic1=[]
dataset_traffic2=[]
dataset_traffic3=[]
predicted_temp=[]
look_back = 90
predictSampleNum = 10

#模型载入
model_node1 = load_model('node1' +'2L' + '.h5')
model_node2 = load_model('node2' +'2L' + '.h5')
model_node3 = load_model('node3' +'2L' + '.h5')
model_node4 = load_model('node4' +'2L' + '.h5')
model_node5 = load_model('node5' +'2L' + '.h5')
model_node6 = load_model('node6' +'2L' + '.h5')
model_traffic1 = load_model('traffic1' +'2L' + '.h5')
model_traffic2 = load_model('traffic2' +'2L' + '.h5')
model_traffic3 = load_model('traffic3' +'2L' + '.h5')

#，装入九十个窗口数据，预测前十个数据

data = pd.read_csv('node1' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node1.append(data[k][0])
dataset_node1 = list(np.array(dataset_node1)/100.0)     
dataset_node1 = [dataset_node1[0:90]]  
dataset_node1 = np.array(dataset_node1)
test_inputdata = np.reshape(dataset_node1, (dataset_node1.shape[0], 1, dataset_node1.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node1 = model_node1.predict(test_temp)
    predicted_temp = predicted_temp + list(predict_node1[0, :])
average_node1=averagenum(predict_node1)
print("node1 第 1次 流量的预测均值：%f" % (average_node1)) 

data = pd.read_csv('node2' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node2.append(data[k][0])
dataset_node2 = list(np.array(dataset_node2)/100.0)     
dataset_node2 = [dataset_node2[0:90]]  
dataset_node2 = np.array(dataset_node2)
test_inputdata = np.reshape(dataset_node2, (dataset_node2.shape[0], 1, dataset_node2.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node2 = model_node2.predict(test_temp)
#    predicted_temp = predicted_temp + list(predict_node1[0, :])
average_node2=averagenum(predict_node2)
print("node2 第 1次 流量的预测均值：%f" % (average_node2)) 


data = pd.read_csv('node3' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node3.append(data[k][0])
dataset_node3 = list(np.array(dataset_node3)/100.0)     
dataset_node3 = [dataset_node3[0:90]]  
dataset_node3 = np.array(dataset_node3)
test_inputdata = np.reshape(dataset_node3, (dataset_node3.shape[0], 1, dataset_node3.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node3 = model_node3.predict(test_temp)
#    predicted_temp = predicted_temp + list(predict_node1[0, :])
average_node3=averagenum(predict_node3)
print("node3 第 1次 流量的预测均值：%f" % (average_node3)) 


data = pd.read_csv('node4' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node4.append(data[k][0])
dataset_node4 = list(np.array(dataset_node4)/100.0)     
dataset_node4 = [dataset_node4[0:90]]  
dataset_node4 = np.array(dataset_node4)
test_inputdata = np.reshape(dataset_node4, (dataset_node4.shape[0], 1, dataset_node4.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node4 = model_node4.predict(test_temp)
#    predicted_temp = predicted_temp + list(predict_node1[0, :])
average_node4=averagenum(predict_node4)
print("node4 第 1次 流量的预测均值：%f" % (average_node4)) 


data = pd.read_csv('node5' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node5.append(data[k][0])
dataset_node5 = list(np.array(dataset_node5)/100.0)     
dataset_node5 = [dataset_node5[0:90]]  
dataset_node5 = np.array(dataset_node5)
test_inputdata = np.reshape(dataset_node5, (dataset_node5.shape[0], 1, dataset_node5.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node5 = model_node5.predict(test_temp)
average_node5=averagenum(predict_node5)
print("node5 第 1次 流量的预测均值：%f" % (average_node5)) 


data = pd.read_csv('node6' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_node6.append(data[k][0])
dataset_node6 = list(np.array(dataset_node6)/100.0)     
dataset_node6 = [dataset_node6[0:90]]  
dataset_node6 = np.array(dataset_node6)
test_inputdata = np.reshape(dataset_node6, (dataset_node6.shape[0], 1, dataset_node6.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_node6 = model_node6.predict(test_temp)
average_node6=averagenum(predict_node6)
print("node6 第 1次 流量的预测均值：%f" % (average_node6)) 

data = pd.read_csv('traffic1' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_traffic1.append(data[k][0])
dataset_traffic1 = list(np.array(dataset_traffic1)/36000000)     
dataset_traffic1 = [dataset_traffic1[0:90]]  
dataset_traffic1 = np.array(dataset_traffic1)
test_inputdata = np.reshape(dataset_traffic1, (dataset_traffic1.shape[0], 1, dataset_traffic1.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_traffic1 = model_traffic1.predict(test_temp)
average_traffic1=averagenum(predict_traffic1)
print("traffic1 第 1次 流量的预测均值：%f" % (average_traffic1))

data = pd.read_csv('traffic2' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_traffic2.append(data[k][0])
dataset_traffic2 = list(np.array(dataset_traffic2)/36000000)     
dataset_traffic2 = [dataset_traffic2[0:90]]  
dataset_traffic2 = np.array(dataset_traffic2)
test_inputdata = np.reshape(dataset_traffic2, (dataset_traffic2.shape[0], 1, dataset_traffic2.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_traffic2 = model_traffic2.predict(test_temp)
average_traffic2=averagenum(predict_traffic2)
print("traffic2 第 1次 流量的预测均值：%f" % (average_traffic2))

data = pd.read_csv('traffic3' + '.csv', usecols=[1])
data = data.values.tolist() 
for k in range(len(data)):
    dataset_traffic3.append(data[k][0])
dataset_traffic3 = list(np.array(dataset_traffic3)/36000000)     
dataset_traffic3 = [dataset_traffic3[0:90]]  
dataset_traffic3 = np.array(dataset_traffic3)
test_inputdata = np.reshape(dataset_traffic3, (dataset_traffic3.shape[0], 1, dataset_traffic3.shape[1]))
for k in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[k, :, :]])
    predict_traffic3 = model_traffic3.predict(test_temp)
average_traffic3=averagenum(predict_traffic3)
print("traffic3 第 1次 流量的预测均值：%f" % (average_traffic3))

#建立socket通信
host = '192.168.108.222'
port = 23455
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('192.168.108.222',23455))

while(1):
#发送数据
    send_list={'node1':average_node1, 'node2':average_node2, 'node3':average_node3, 'node4':average_node4, 'node5':average_node5, 'node6':average_node6, 'traffic1':average_traffic1, 'traffic2':average_traffic2, 'traffic3':average_traffic3}
    datasend = json.dumps(send_list)
    sock.send(datasend.encode('utf-8'))

#接受数据
    datarecv=sock.recv(1024)
    datarecv = datarecv.decode('utf-8')
    datarecv = json.loads(datarecv)
    
#数据分解
    datanode1 = datarecv['node1'] 
    datanode2 = datarecv['node2'] 
    datanode3 = datarecv['node3'] 
    datanode4 = datarecv['node4'] 
    datanode5 = datarecv['node5'] 
    datanode6 = datarecv['node6'] 
    datatraffic1 = datarecv['traffic1'] 
    datatraffic2 = datarecv['traffic2'] 
    datatraffic3 = datarecv['traffic3']

#不断移位    
    for i in len(datanode1):
        dataset_node1.append(datanode1[i])
    dataset_node1 = [dataset_node1[10:]]
    dataset_node1 = np.array(dataset_node1)
    test_inputdata = np.reshape(dataset_node1, (dataset_node1.shape[0], 1, dataset_node1.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node1 = model_node1.predict(test_temp)
        
    average_node1=averagenum(predict_node1)
    
    for i in len(datanode2):
        dataset_node2.append(datanode2[i])
    dataset_node2 = [dataset_node2[10:]]
    dataset_node2 = np.array(dataset_node2)
    test_inputdata = np.reshape(dataset_node2, (dataset_node2.shape[0], 1, dataset_node2.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node2 = model_node2.predict(test_temp)
      
    average_node2=averagenum(predict_node2)    
        
    for i in len(datanode3):
        dataset_node3.append(datanode3[i])
    dataset_node3 = [dataset_node3[10:]]
    dataset_node3 = np.array(dataset_node3)
    test_inputdata = np.reshape(dataset_node3, (dataset_node3.shape[0], 1, dataset_node3.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node3 = model_node3.predict(test_temp)
      
    average_node3=averagenum(predict_node3)
    
    for i in len(datanode4):
        dataset_node4.append(datanode4[i])
    dataset_node4 = [dataset_node4[10:]]
    dataset_node4 = np.array(dataset_node4)
    test_inputdata = np.reshape(dataset_node4, (dataset_node4.shape[0], 1, dataset_node4.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node4 = model_node4.predict(test_temp)
      
    average_node4=averagenum(predict_node4)
        
    for i in len(datanode5):
        dataset_node5.append(datanode5[i])
    dataset_node5 = [dataset_node5[10:]]
    dataset_node5 = np.array(dataset_node5)
    test_inputdata = np.reshape(dataset_node5, (dataset_node5.shape[0], 1, dataset_node5.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node5 = model_node5.predict(test_temp)
      
    average_node5=averagenum(predict_node5)       
    
    for i in len(datanode6):
        dataset_node6.append(datanode6[i])
    dataset_node6 = [dataset_node6[10:]]
    dataset_node6 = np.array(dataset_node6)
    test_inputdata = np.reshape(dataset_node6, (dataset_node6.shape[0], 1, dataset_node6.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_node6 = model_node6.predict(test_temp)
      
    average_node6=averagenum(predict_node6) 
    
    for i in len(datatraffic1):
        dataset_traffic1.append(datatraffic1[i])
    dataset_traffic1 = [dataset_traffic1[10:]]
    dataset_traffic1 = np.array(dataset_traffic1)
    test_inputdata = np.reshape(dataset_traffic1, (dataset_traffic1.shape[0], 1, dataset_traffic1.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_traffic1 = model_traffic1.predict(test_temp)
      
    average_traffic1=averagenum(predict_traffic1) 
    
    for i in len(datatraffic2):
        dataset_traffic2.append(datatraffic2[i])
    dataset_traffic2 = [dataset_traffic2[10:]]
    dataset_traffic2 = np.array(dataset_traffic2)
    test_inputdata = np.reshape(dataset_traffic2, (dataset_traffic2.shape[0], 1, dataset_traffic2.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_traffic2 = model_traffic2.predict(test_temp)
      
    average_traffic2=averagenum(predict_traffic2) 
    
    for i in len(datatraffic3):
        dataset_traffic3.append(datatraffic3[i])
    dataset_traffic3 = [dataset_traffic3[10:]]
    dataset_traffic3 = np.array(dataset_traffic3)
    test_inputdata = np.reshape(dataset_traffic3, (dataset_traffic3.shape[0], 1, dataset_traffic3.shape[1]))
    for k in range(test_inputdata.shape[0]):
        test_temp = np.array([test_inputdata[k, :, :]])
        predict_traffic3 = model_traffic3.predict(test_temp)
      
    average_traffic3=averagenum(predict_traffic3)
    
    







