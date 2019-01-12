import matplotlib.pyplot as plt
# import csv
import xlrd
import numpy as np
# import tensorflow as tf
# import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from keras.models import load_model
# import time

pattern = '虚拟机1到虚拟机2流量'
xlsx_file_name =pattern + '.xlsx'
trainingSampleNum = 20000
testSampleNum = 7000
look_back = 12*10
predictSampleNum = 12
model = load_model(pattern +'N1toN2' + '.h5')

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def creat_testdata(data):
    test_input = []
    for i in range((len(data)-look_back)//predictSampleNum):
        test_input.append(data[predictSampleNum*i:predictSampleNum*i+look_back])
    return np.array(test_input)


testDataSet = xlrd.open_workbook(xlsx_file_name).sheet_by_index(0).col_values(1)[23000:27000]
testDataSet = list(np.array(testDataSet)/116360262.0)
test_inputdata = creat_testdata(testDataSet)
test_inputdata = np.reshape(test_inputdata, (test_inputdata.shape[0], 1, test_inputdata.shape[1]))
#print(testDataSet[1:300])

predicted_data = []
predicted_temp = []
for i in range(test_inputdata.shape[0]):
    test_temp = np.array([test_inputdata[i, :, :]])
    Predict = model.predict(test_temp)
    predicted_temp = predicted_temp + list(Predict[0, :])


actual = testDataSet[look_back:look_back-1+len(predicted_temp)]
relativeError = []
for i in range(len(actual)):
    if(actual[i] > 0.1):
        relativeError.append(abs(actual[i] - predicted_temp[i]))
avgRelativeErr = np.mean(relativeError)
print(avgRelativeErr)

prediction = open("prediction_results.dat", "w+")
for elements in predicted_temp:
    prediction.write(str(elements)+'\n')
actualfile = open("actual_results.dat", "w+")
for elements in actual:
    actualfile.write(str(elements)+'\n')
prediction.close()
actualfile.close()

errorFile = open("error.dat", "w+")
for elements in relativeError:
    errorFile.write(str(elements)+'\n')
errorFile.close()


plt.plot(predicted_temp[0:3300], label='predicted')
plt.plot(actual[0:3300], label='true')
plt.legend()
plt.savefig('traffic.png')
plt.show()

plt.plot(actual[0:3300], label='true',color='red')
plt.legend()
plt.savefig('traffic1.png')
plt.show()

plt.plot(predicted_temp[0:3300], label='predicted')
#plt.plot(actual[0:3300], label='true')
plt.legend()
plt.savefig('traffic2.png')
plt.show()

n = plt.hist(np.array(relativeError), bins=20, range=(0, 0.2))
plt.savefig('relativeError.png')
plt.xticks(np.arange(0, 0.2, 0.02))
print(np.sum(n[0]))

plt.show()

