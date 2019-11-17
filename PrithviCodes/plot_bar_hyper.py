



import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.datasets import make_classification



d = {}
d[300] = 0.8557981728020356
d[100] = 0.5301725328598704
d[50] = 0.35620644263291756

gam = {}
gam[300] = "gamma: 1e-9"    # changed
gam[100] = "gamma: 1e-09"
gam[50] = "gamma: 1e-07"

C = {}
C[300] = "C: 1"
C[100] = "C: 10"
C[50] = "C: 10"

acc = {}

acc[300] = 0.8770370370370371
acc[100] = 0.6044444444444445
acc[50] = 0.43555555555555553




li = []



for k,v in d.items():
	ze = k
	tup = (ze, v)
	li.append(tup)
print(li)
# np.save("yBinned.npy", ynew)
# print((np.unique(ynew)).size)


#plt.bar(range(len(li)), [val[1] for val in li], align='center')
ax = plt.subplot(111)
w = 0.3
ind = np.arange(3) 
plt.bar( ind , [val[1] for val in li], width=w, color='b', align='center', label='F1-score')
plt.bar( ind+w, [acc[k] for k in acc], width=w, color='g', align='center', label='Accuracy')


plt.xticks(range(len(li)), [str(val[0])+"\n"+gam[val[0]]+"\n"+C[val[0]] for val in li])
#plt.xticks(rotation=70)

#labels
plt.title("Bin Sizes(in $ millions) VS F1-score/Accuracy for SVM")
plt.ylabel('F1-score/Accuracy')
#plt.tick_params(axis='x', which='major', pad=0)
#ax = plt.subplot()
ax.set_xlabel("Bin Sizes ($ Million)", labelpad=30)
#plt.xlabel('Revenue $ (Million)', 30)
plt.tight_layout()
plt.legend()
# plt.show()

plt.savefig('BinVSF1SVM_2.png')







'''
ax = plt.subplot(111)
w = 0.3
ax.bar(x-w, y, width=w, color='b', align='center')
ax.bar(x, z, width=w, color='g', align='center')
ax.bar(x+w, k, width=w, color='r', align='center')
ax.xaxis_date()
ax.autoscale(tight=True)

plt.show()

'''