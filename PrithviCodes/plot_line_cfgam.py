





import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

C300 = np.load('C300.npy')
f_gam = np.load('Gammas300.npy')
# xval = np.load('Xvalues.npy')

gamma_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001,0.01,0.1,1,10,100,1000]
orig_gamma_list = ['1e-9', '1e-8', '1e-7', '1e-6', '1e-5', '0.0001', '0.001','0.01','0.1','1','10','100','1000']
gamma_len = list(range(13))

C_list = [0.01, 0.1, 1, 10 , 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
orig_C_list = ['0.01', '0.1', '1', '10' , '1e+2', '1e+3', '1e+4', '1e+5', '1e+6', '1e+7', '1e+8', '1e+9', '1e+10']


C_list = list(range(len(C_list)))
#fscore = [0.01, 0.1, 1,10 , 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]

#C300 = [0.8062, 0.8064, 0.8067]
#fscore = [0.01, 0.1, 0.5]

print(C300)
print(f_gam)
print(C_list)
print(gamma_list)

#print(gam)
#print(len(gam))
#print(fscore)
#print(len(fscore))
#print(gamma_list)
#print(len(gamma_list))


#x = range(ytest.size)


# '''
# f1 score vs c hyperparameters 
fig, ax = plt.subplots()

ax.plot(C_list, list(C300))

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

plt.legend(loc='upper left');

#plt.yticks(range(len(ypredSorted)), [val for val in ypredSorted])

plt.title("F1-score VS C hyperparameter")
plt.xticks(range(13), orig_C_list, rotation=45)
plt.xlabel("Parameter: C")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig('Line_chart_C_f1score.png')
#plt.show()
# '''
plt.clf()


'''
# for gamma

fig, ax = plt.subplots()

ax.plot(gamma_len, list(f_gam))


ax.grid()

plt.legend(loc='upper right');

#plt.yticks(range(len(ypredSorted)), [val for val in ypredSorted])

plt.title("F1-score VS Gamma hyperparameter")
plt.xticks(range(13), orig_gamma_list, rotation=45)
plt.xlabel("Parameter: Gamma")
plt.ylabel("F1-score")
plt.tight_layout()
plt.savefig('Line_chart_gamma_f1score.png')
plt.show()

# '''