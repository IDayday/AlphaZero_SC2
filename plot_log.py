
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import json
import ast
import matplotlib.pyplot as plt

log = []
with open('evaluatelog_sample_vs_max.txt',mode='r') as f:
    lines = f.readlines()
    for i in lines:
        i.strip()
        dic = ast.literal_eval(i)
        log.append(dic)
        # print(dic)
        # print(type(dic))


# name = 'model_{}.pt_vs_model_{}.pt'.format(my, op)

total = len(log)
colum = total//9
log_array = np.zeros((9,9))
flag = 0
i = 0
j = 0
for ind, line in enumerate(log):
    if (ind+1)%colum == 0:
        for vs, win_rate in line.items():
            if ((flag+1)%colum) == 0:
                log_array[i,j] = win_rate
                j += 1
            flag += 1
        i += 1
        j = 0

print(log_array)


# np.random.seed(20180316)
# x = np.random.randn(4, 4)
 
f, ax1 = plt.subplots(figsize=(9,9),nrows=1)

sns.heatmap(log_array, annot=True, ax=ax1).invert_yaxis()
ax1.set_xlabel("checkpoints",fontsize=15)
ax1.set_ylabel("checkpoints",fontsize=15)
# plt.savefig('./eva_result.eps')
plt.savefig('./eva_sample_vs_max_result.jpg')