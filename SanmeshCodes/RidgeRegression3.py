#START#############################################
#########################################################
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt


#INPUTS#############################################
#########################################################
loadRawXFromNumpy = 1
loadTrainAndTestData = 1
dataset_X_reimported = pd.read_csv('xgFeatures_156.csv')
numpyUrl = 'ReducedRawData/pcaOutputScalingNum99Perc.npy'
##testAndTrainName = "pcaOutputScalingNum99Perc.npy"
testAndTrainName = "pcaOutputScalingNum99Perc.npy"
#loadData#############################################
#########################################################
from sklearn import linear_model
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset_X_reimported = pd.read_csv('xgFeatures_156.csv')
dataset_y_reimported = pd.read_csv('Encoded_y_revenue.csv') #or - rating.csv
dataset_X_reimported.rename(columns={'5':'Film runtime',
                                     '0':'Film budget',
                                     '4':'Day of year',
                                     '7':'Year of release',}, 
                 inplace=True)
dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here
if loadRawXFromNumpy ==1:
        X = np.load(numpyUrl)
else:
        X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values

def calc_rmse(pred, label): 
	return np.sqrt(np.mean((pred-label)**2))

def normalized_rmse(pred,label):
    rmse=calc_rmse(pred,label)
    norm_rmse=rmse/(np.max(label)-np.min(label))
    return norm_rmse

def find_best_alpha(X,y):
	ridgecv = RidgeCV(alphas=[0.001, 0.1, 0.5, 1, 5, 10],normalize = True).fit(X, y)
	best_alpha=ridgecv.alpha_
	return best_alpha

#Calculate Alpha#############################################
#########################################################
if loadTrainAndTestData == 1:
##        print("TrainAndTestData/xtrain_" + testAndTrainName)
        xtrain = np.load("TrainAndTestData/xtrain_" + testAndTrainName)
        xtest = np.load("TrainAndTestData/xtest_" + testAndTrainName)
        ytrain = np.load("TrainAndTestData/ytrain_" + testAndTrainName)
        ytest = np.load("TrainAndTestData/ytest_" + testAndTrainName)
else:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)#,shuffle=False)

alpha=find_best_alpha(xtrain,ytrain)
print("Alpha: {}".format(alpha))
reg = Ridge(alpha=alpha,normalize=True)
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
#rmse=mean_squared_error(ytest, ypred)
rmse=calc_rmse(ypred,ytest)
norm_rmse=normalized_rmse(ypred,ytest)
##np.savetxt('ypredRidgeRegWithPCA.csv', ypred, delimiter=',')
##np.savetxt('ytestRidgeRegWithPCA.csv', ytest, delimiter=',')
print("RMSE: {}".format(rmse))
print("Normalized RMSE: {}".format(norm_rmse))
print("R2 score" ,r2_score(ytest, ypred))

x = range(ytest.size)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, ytest, s=10, c='b', marker="s", label='test')
ax1.scatter(x,ypred, s=10, c='r', marker="o", label='predicted')
##plt.plot((ytest+ypred)/2)
plt.errorbar(x, (ytest+ypred)/2, yerr=np.abs(ytest-ypred)/2, xlolims=True, label='error bar', fmt = ',')
plt.legend(loc='upper right');

##############plot 2
sortedIndices = np.argsort(ytest)
##print(sortedIndices)
ytestSorted = ytest[sortedIndices]
ypredSorted = ypred[sortedIndices]

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.scatter(x, ytestSorted, s=10, c='b', marker="s", label='Test Actual Revenue')
ax2.scatter(x,ypredSorted, s=10, c='r', marker="o", label='Predicted Revenue')
##plt.plot((ytest+ypred)/2)
plt.errorbar(x, (ytestSorted+ypredSorted)/2, yerr=np.abs(ytestSorted-ypredSorted)/2, xlolims=True, label='error bar', fmt = ',')
plt.legend(loc='upper left');
plt.title("Test Actual vs Predicted Revenue, sorted by Actual Revenue")
plt.xlabel("Particular x data point from PCA 99")
plt.ylabel("Revenue")


plt.show()
