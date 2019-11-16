




import numpy as np
import matplotlib.pyplot as plt

import pandas as pd



dataset_y_reimported = pd.read_csv('Encoded_y - revenue.csv')



dataset_X_reimported = pd.read_csv('Encoded_X.csv')

'''dataset_X_reimported.rename(columns={'5':'Film runtime',
                                     '0':'Film budget',
                                     '4':'Day of year',
                                     '7':'Year of release',}, 
                 inplace=True)
'''
 

dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here


#X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values

binWindowSize = 100*(10**6)
print(binWindowSize)
##ynew = np.floor(y/binWindowSize, casting='int',)
ynew = (y/binWindowSize).astype(int)
print("y", y)
print(len(y))
print("ynew", ynew)
print(len(ynew))

d = {}
s = set(ynew)
for i in s:
	d[i] = 0

for i in ynew:
	d[i] += 1


li = []
binWindowSize /= 10**8
for k,v in d.items():
	ze = k*binWindowSize+binWindowSize
	tup = (ze, v)
	li.append(tup)
print(li)
# np.save("yBinned.npy", ynew)
# print((np.unique(ynew)).size)


plt.bar(range(len(li)), [val[1] for val in li], align='center')
plt.xticks(range(len(li)), [100*val[0] for val in li])
plt.xticks(rotation=70)

#labels
plt.title("Distribution of Revenue (in $ millions) VS count of movies")
plt.ylabel('No. of movies')
plt.xlabel('Revenue $ (Million)')
plt.tight_layout()
#plt.show()
plt.savefig('RevenueVSCount.png')
# plot



#MY PLOT
'''

labels, ys = zip(*li)
print("Labels :",labels)
xs = np.array(labels)# np.arange(len(labels)) 
width = 1

fig = plt.figure()                                                               
ax = fig.gca()  #get current axes
ax.bar(xs, ys, align='edge', width=0.8)

plt.title("Distribution of Revenue (in 100 million) VS count of movies")
plt.xlabel('Revenue $       (10^8)')
plt.ylabel('No. of movies')
ax.set_xticks(np.arange(len(labels)))
#ax.locator_params(nbins=10, axis='x')
index = np.arange(len(labels))
#plt.xticks(index, labels, fontsize=5, rotation=30)
ax.set_xticklabels(labels, rotation="45", ha="right")
#plt.tight_layout()
#plt.show()
plt.savefig('RevenueVSCount.png')
'''




