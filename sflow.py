# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:48:00 2018

@author: 28676
"""

import requests
import json
import time
from threading import Timer
def bytes2human(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i+1)*10
 
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n)/prefix[s]
            return '%s%s' % (value, s)
    return '%sK' % n

flow = {'keys':'ipsource,ipdestination',
 'value':'bytes','log':True}
requests.put('http://192.168.108.222:8008/flow/tcp/json',data=json.dumps(flow))
requests.put('http://192.168.108.222:8008/flow/udp/json',data=json.dumps(flow))
flowurl = 'http://192.168.108.222:8008/activeflows/ALL/tcp/json'
flowID = -1
vm=["10.10.10.12","10.10.10.7","10.10.10.9"]
a=0
file=open('1.txt',mode='a')
#def getrecord()
  #r = requests.get(flowurl + "&flowID=" + str(flowID))
  #if r.status_code != 200: break
for i in range(30):
    r = requests.get(flowurl)
    flows = r.json()
    print(vm[1]+','+vm[2])
    for i in range(len(flows)):
        if(flows[i]['key']==vm[1]+','+vm[2]):
            traffic=flows[i]['value']
            a+=traffic
    time.sleep(1)
    print(bytes2human(traffic))
    file.write(str(bytes2human(traffic))+'\n')
    print(bytes2human(a))
file.close()