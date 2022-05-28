import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(20180316)
x = np.random.randn(4, 4)
 
f, (ax1, ax2) = plt.subplots(figsize=(6,6),nrows=2)
 
sns.heatmap(x, annot=True, ax=ax1)
sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
plt.savefig('./eva_result.eps')