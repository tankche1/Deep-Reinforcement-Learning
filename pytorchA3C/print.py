import os
import re
import string

File=open(os.getcwd()+'/record.txt','r')
Text=File.read()
#print(A)

A=re.findall(r'Time (.*),episode reward',Text)
time=[]
for B in A:
    #print(B)
    time.append(int(B))
print(time)
rewards=[]

A=re.findall(r'episode reward (.*), episode length',Text)
#print(A)
for B in A:
    #print(B)
    rewards.append(string.atof(B))
print(rewards)
#B=re.match(r'Time (.*)h',A)
#print(B.group(1))


import numpy as np
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(time, rewards)
plt.xlabel('time/min')
plt.ylabel('score')
plt.show()
