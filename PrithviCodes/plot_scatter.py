import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# for 300 intervalls 
'''
ytest = np.load('actualvalues.npy')
ypred = np.load('predictedvalues.npy')
'''
# for 100 intervals
#'''
ytest = np.load('actualvalues_100interval_SVM_Correct.npy')
ypred = np.load('predictedvalues_100interval_SVM_Correct.npy')
#'''

# xval = np.load('Xvalues.npy')


sortedIndices = np.argsort(ytest)
##print(sortedIndices)
ytestSorted = ytest[sortedIndices]
ypredSorted = ypred[sortedIndices]


binW = 100   #change

ytestSorted = ytestSorted*binW+binW
ypredSorted = ypredSorted*binW+binW



x = range(ytest.size)
fig = plt.figure()
ax2 = fig.add_subplot(111)

ax2.scatter(x, ytestSorted, s=10, c='b', marker="s", label='Actual Revenue')
ax2.scatter(x,ypredSorted, s=10, c='g', marker="o", label='Predicted Revenue')
##plt.plot((ytest+ypred)/2)
plt.errorbar(x, (ytestSorted+ypredSorted)/2, yerr=np.abs(ytestSorted-ypredSorted)/2, xlolims=True, label='error bar', fmt = ',', linewidth=0.3, color='red')
plt.legend(loc='upper left');

#plt.yticks(range(len(ypredSorted)), [val for val in ypredSorted])

plt.title("Actual Revenue vs Predicted Revenue (SVM) Bin Size: 100")
plt.xlabel("Particular x data point")
plt.ylabel("Revenue (Million $)")

plt.savefig('Scatter_predVSActual100.png')
plt.show()