





import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

ytest = np.load('actualvalues.npy')
ypred = np.load('predictedvalues.npy')
# xval = np.load('Xvalues.npy')


sortedIndices = np.argsort(ytest)
##print(sortedIndices)
ytestSorted = ytest[sortedIndices]
ypredSorted = ypred[sortedIndices]


binW = 300

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

plt.title("Actual Revenue vs Predicted Revenue, sorted by Actual Revenue (SVM)")
plt.xlabel("Particular x data point")
plt.ylabel("Revenue (Million $)")

plt.savefig('Scatter_predVSActual.png')
#plt.show()