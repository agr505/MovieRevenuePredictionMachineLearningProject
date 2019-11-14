#START#############################################
#########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#INPUTS#############################################
#########################################################
loadRawXFromNumpy = 1
loadTrainAndTestData = 1
dataset_X_reimported = pd.read_csv('xgFeatures_156.csv')
numpyUrl = 'ReducedData/pcaOutputScalingNum99Perc.npy'

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

def rmse(pred, label): 
	return np.sqrt(np.mean((pred-label)**2))
def find_best_alpha(X,y):
	ridgecv = RidgeCV(alphas=[0.001, 0.1, 0.5, 1, 5, 10],normalize = True).fit(X, y)
	best_alpha=ridgecv.alpha_
	return best_alpha

#Calculate Alpha#############################################
#########################################################
if loadTrainAndTestData == 1:
        xtrain = np.load("ReducedData/xtrain_xgFeatures_156.npy")
        xtest = np.load("ReducedData/xtest_xgFeatures_156.npy")
        ytrain = np.load("ReducedData/ytrain_xgFeatures_156.npy")
        ytest = np.load("ReducedData/ytest_xgFeatures_156.npy")
else:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

alpha=find_best_alpha(xtrain,ytrain)
print("Alpha: {}".format(alpha))
reg = Ridge(alpha=alpha,normalize=True)
reg.fit(xtrain,ytrain)
ypred=reg.predict(xtest)
print("***")
print(ypred[0])
print(ytest[0])
print(ytest[0]-ypred[0])
print("***")
#rmse=mean_squared_error(ytest, ypred)
rmse=rmse(ypred,ytest)
np.savetxt('ypredRidgeRegWithPCA.csv', ypred, delimiter=',')
np.savetxt('ytestRidgeRegWithPCA.csv', ytest, delimiter=',')
print("RMSE: {}".format(rmse))
print("R2 score" ,r2_score(ytest, ypred))

