# a = []
# for i, act in enumerate(a):
#     print(i)
#     print(act)


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
b = []
c = [[1,2,3],[4,2,6]]
a.extend(b)
a.extend(c)
# p = [x-1 for x in a]
# p.pop(0)
print(a)