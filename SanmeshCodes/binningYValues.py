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

binWindowSize = 300*(10**6)
##ynew = np.floor(y/binWindowSize, casting='int',)
ynew = (y/binWindowSize).astype(int)
print("y", y)
print("ynew", ynew)
np.save("yBinned.npy", ynew)
print((np.unique(ynew)).size)
