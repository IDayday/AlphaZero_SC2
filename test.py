# a = []
# for i, act in enumerate(a):
#     print(i)
#     print(act)

from cmath import inf
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

# a = [1,2,3,1,10]
# print(a[:-1])
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

# import torch
# a = inf
# if a == float("inf"):
#     print('yes')
# a = torch.randn((5,1))
# b = a.sum()
# c = a/b
# print(a)
# print(b)
# print(c)
# print(c.sum())

# probs = [0.1,0.2,0.1,0.3,0.2,0.2,0.1]
# p = np.random.dirichlet(0.3 * np.ones(len(probs)))
# print(p.sum())

# import numpy as np

# s = 0
# for k in range(10000):
#     a =  np.random.randint(0,100000,128)
#     # print(len(a))
#     c = 0
#     for i in range(len(a)):
#         for j in range(i+1,len(a)):
#             if abs(a[i]-a[j])<=5:
#                 c += 1
#     s += c
# print(s)
# import torch
# import numpy as np
# act = [0,4,6,8,10]
# a = torch.tensor([0.1,0.2,0.5,0.1,0.1])
# b = np.random.choice(act,p=a)

# print(b)

# import multiprocessing
# import time
# def func(msg):
#   for i in range(3):
#     print (msg)
#     time.sleep(1)
#   return "done " + msg
# if __name__ == "__main__":
#     pool = multiprocessing.Pool(processes=4)
#     result = []
#     for i in range(10):
#         msg = "hello %d" %(i)
#         result.append(pool.apply_async(func, (msg, )))
#     pool.close()
#     pool.join()
#     for res in result:
#         print (res.get())
#         print ("Sub-process(es) done.")
x = [1,2,3]
y = [2,3,4]
a = [x, y]
c, d = a
print(c)
print(d)