# a = []
# for i, act in enumerate(a):
#     print(i)
#     print(act)

import torch
import numpy as np

# a = np.array([0,1,2,3,4,5])
# if 5 in a:
#     print('yes')

# a = [[1,2],[10,9]]
# b = []
# for p in a:
#     p = [x-1 for x in p]
#     b.append(p)
# print(b)

a = [1,2,3,1,10]
print(a[:-1])
# b = []
# c = [[1,2,3],[4,2,6]]
# a.extend(b)
# a.extend(c)
# p = [x-1 for x in a]
# p.pop(0)
# print(a)

# from collections import deque
# import random
# a = [[1,2,3],(4,5,6),(7,8,9)]
# c = [(3,3,3),[1,0,2]]
# b = []
# b.extend(a)
# b.extend(c)
# d = deque(maxlen=1000)
# d.extend(b)
# minibatch = random.sample(d,2)
# print(minibatch)
# i=1
# loss =0
# print(f"epoch{i} batchloss: ",loss)
# available_actions = [0,2]
# action_prob = torch.tensor([[0.1703, 0.1612, 0.2868, 0.1987, 0.1831]])
# mask = [action_prob[0][i] for i in available_actions]
# print(mask)

# a = torch.zeros((5,1)).to("cuda")
# b = 1
# c = torch.tensor([18])

# a[0] = b
# a[1] = c
# print(c)
# print(a)

