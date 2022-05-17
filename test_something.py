# from collections import namedtuple
# from telnetlib import BM
import torch


# BMState = namedtuple('BMState', ['n_scv', 'n_marine', 'n_depot', 'n_barracks',
#                                  'mineral', 'food_used', 'food_cap', 'food_workers', 'army_count'])

# state_1 = BMState(1,2,3,4,5,6,7,8,9)
# state_2 = BMState(9,8,7,6,5,4,3,2,1) * -1
# print(state_2)
# state = state_1 + state_2
# print(state)

a = torch.randn((1,5))
b = [0,1]
c = [a[0][i] for i in b]
print(c)