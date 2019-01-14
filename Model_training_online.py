 # import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import xlrd
import numpy as np
import tensorflow as tf
# import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
start = time.clock()
predicted_temp=[]
trainnum = 1
temp = 0
real=[]

from keras.models import load_model
# file location
#pattern = '虚拟机1到虚拟机2流量'
#xlsx_file_name =   pattern + '.xlsx'
#trainingSampleNum = 20000
#testSampleNum = 7000
look_back = 12*10
predictSampleNum = 12
data = []
dataset = []
useflags = [0 for i in range(100)] #标记文件是否被读入

# # create model
model = Sequential()
# model.add(LSTM(64, return_sequences=True, input_shape=(1, look_back)))
# model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, input_shape=(1, look_back), return_sequences=True))
# model.add(LSTM(512, return_sequences=True))

model.add(LSTM(256))

# model.add(LSTM(1024, return_sequences=True))

model.add(Dense(predictSampleNum, activation='relu'))
 
def creat_dataset(data, look_back):
    dataX, dataY=[], []
    for i in range(len(data)-look_back-predictSampleNum - 1):
        dataX.append(data[i:(i+look_back)])
        dataY.append(data[(i+look_back):(i+look_back + predictSampleNum)])
        # dataY.append(data[(i + look_back)])
    return np.array(dataX), np.array(dataY)


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def creat_testdata(data):      #预测数据
    test_input = []
    for i in range((len(data)-look_back)//predictSampleNum):
        test_input.append(data[predictSampleNum*i:predictSampleNum*i+look_back])
    return np.array(test_input)

#读入CSV数据
while(1): 
    for i in range(temp, 100):
        
        if(os.path.exists(str(i) + '.csv') and (useflags[i] == 0)):
            trainnum = 0
            useflags[i] = 1
            if(i <= 10):
                data = pd.read_csv(str(i) + '.csv', usecols=[1])
                data = data.values.tolist() 
                for k in range(len(data)):
                    dataset.append(data[k][0])
                if(i == 10):
                    break
            elif(i > 10): 
                predicted_temp = []                               #i >= 10 进行预测
                testdataset = []
                #dataset=[]
                real=[]
                data = pd.read_csv(str(i-1) + '.csv', usecols=[1])
                data = data.values.tolist() 
                for k in range(len(data)-look_back+1,len(data)):
                    testdataset.append(data[k][0])
                data = pd.read_csv(str(i) + '.csv', usecols=[1])
                data = data.values.tolist() 
                for k in range(0,len(data)):
                    dataset.append(data[k][0])
                    testdataset.append(data[k][0])
                    real.append(data[k][0])
                test_inputdata = creat_testdata(testdataset)
                average1=averagenum(real)/25000000
                test_inputdata = np.reshape(test_inputdata, (test_inputdata.shape[0], 1, test_inputdata.shape[1]))
                for k in range(test_inputdata.shape[0]):
                    test_temp = np.array([test_inputdata[k, :, :]])
                    Predict = model.predict(test_temp)
                    predicted_temp = predicted_temp + list(Predict[0, :])
                average=averagenum(predicted_temp)
                print("第 %d 次 流量的预测均值：%f" % (i,average))
                print("第 %d 次 流量的真实均值：%f" % (i,average1))
                diff = average1 - average
                print("第 %d 次 流量的差值：%f" % (i,diff))
                break
        else:
            break
    
   
     # extract data 'data' store the whole training data set
#data = xlrd.open_workbook(xlsx_file_name).sheet_by_index(0).col_values(1)[4000:3000+trainingSampleNum]
    if(trainnum == 0):
        plt.plot(predicted_temp[0:], label='predicted')
       
        real = list(np.array(real)/25000000.0)
       
        
        plt.plot(real[0:],label='real')
        #plt.plot(actual[0:3300], label='true')
        plt.legend()
        plt.savefig('traffic2.png')
        plt.show()
        temp = i + 1
        trainnum = 1
        dataset = list(np.array(dataset)/25000000.0)
        trainX, trainY = creat_dataset(dataset, look_back)
    #testX, testY = creat_dataset(dataset[trainingSampleNum:trainingSampleNum + testSampleNum], look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='AdaGrad', metrics=['accuracy'])
       # print(model)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        model.fit(trainX, trainY, epochs=1, batch_size=1)
    #    model.save(pattern +'N1toN2' +'.h5')
    #    print('model have saved')
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
