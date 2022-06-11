
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
import ast
import matplotlib.pyplot as plt

logpath = './checkpoint/default_evaluate_sample2/log.txt'
loss = []
vloss = []
ploss = []
with open(logpath, mode='r') as f:
    lines = f.readlines()
    for i,s in enumerate(lines):
        temp_list = s.split(':')
        temp_list[-1] = temp_list[-1].strip()
        if i%3 == 0:
            loss.append(round(float(temp_list[-1]),4)) 
        elif (i-1)%3 == 0:
            vloss.append(round(float(temp_list[-1]),4))
        elif (i-2)%3 == 0:
            ploss.append(round(float(temp_list[-1]),4))
print(loss[:5])   
print(vloss[:5])   
print(ploss[:5])

plt.figure(figsize=(12,8))
ax1=plt.subplot(111)
plt.title('Batch Loss Change',fontsize=20)
# ax1.set_xlim(0,loop_num+1)
# ax1.set_ylim(0,loss_max+1)
# ax1.plot(entropy_data_smooth, c='deepskyblue')
ax1.plot(loss, color = 'blue', linewidth = 1.00)
ax1.plot(vloss, color = 'r', linewidth = 1.00)
ax1.plot(ploss, color = 'g', linewidth = 1.00)
plt.tick_params(labelsize=20)
plt.legend(["total loss",'value loss','policy loss'],prop={'size': 15})
plt.savefig('./sample_trainloss.jpg', dpi=200)