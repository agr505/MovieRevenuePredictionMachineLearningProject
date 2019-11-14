# -*- coding: utf-8 -*-
"""
Created on Sun Aug 4 06:46:53 2019
@author: rian-van-den-ander
"""
##################################################
#Inputs###############
numCompVar = 0.99
doWeScale = 1
#loadPcaData = 1

##################################################
#GETTING DATA###############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_X_reimported = pd.read_csv('Encoded_X.csv')
dataset_y_reimported = pd.read_csv('Encoded_y - revenue.csv') #or - rating.csv

dataset_X_reimported.rename(columns={'5':'Film runtime',
                                     '0':'Film budget',
                                     '4':'Day of year',
                                     '7':'Year of release',}, 
                 inplace=True)
 

dataset_reimported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataset_reimported = dataset_reimported.replace([np.inf, -np.inf], np.nan)
dataset_reimported = dataset_reimported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here


X = dataset_reimported.iloc[:, 1:-2].values
y = dataset_reimported.iloc[:, -1].values
X_names = dataset_reimported.columns[1:-2].values

##################################################
#Feature Scaling###############
from sklearn.preprocessing import StandardScaler
if doWeScale ==1:
    X_Scaled = StandardScaler().fit_transform(X)
else:
    X_Scaled = X

##################################################
#PCA###############
from sklearn.decomposition import PCA
pca = PCA(n_components = numCompVar, svd_solver = 'full')
pca.fit(X_Scaled)
reduced_data = pca.transform(X_Scaled)
np.save("pcaOutput.npy", reduced_data)

##################################################
#Print PCA Output###############
recoveredVarPer = np.sum(pca.explained_variance_ratio_)*100
print("Recovered Variance % = ", recoveredVarPer)
numOfReducFeat = (pca.singular_values_).size
origNumFeat = X.shape[1]
print("original features: ", origNumFeat)
print("Reduced features: ", numOfReducFeat)

plt.style.use('ggplot')
plt.bar(range(numOfReducFeat), pca.explained_variance_ratio_*100, color='green')
plt.xlabel("Principle Component")
plt.ylabel("Recovered Variance %")
plt.title("Recovered Variance % of each principle component")
plt.show()


