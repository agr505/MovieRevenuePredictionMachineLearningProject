#START#############################################
#########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#INPUTS#############################################
#########################################################
loadFromNumpy = 0 #set to 1 if want to load numpy from variable numpyUrl
#otherwise, set 1 to load data from dataset_X_reimported url
dataset_X_reimported = pd.read_csv('xgFeatures_156.csv')
numpyUrl = 'ReducedData/pcaOutputScalingNum99Perc.npy'
percentageOfTest = 0.2

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
if loadFromNumpy ==1:
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

#loadData#############################################
#########################################################
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=percentageOfTest)
np.save("xtrain.npy", xtrain)
np.save("xtest.npy", xtest)
np.save("ytrain.npy", ytrain)
np.save("ytest.npy", ytest)
print("train data Size = ",xtrain.shape)
print("train data Size = ", xtest.shape)
