# -*- coding: utf-8 -*-
"""
Created on Sun Aug 4 06:46:53 2019
@author: rian-van-den-ander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

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

#NO TRAIN AND TEST SPLIT as I need to link back to features


regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
                         subsample = 0.8, objective='reg:squarederror')
regressor.fit(X, y)

# analysis: which are the most important features?
sorted_importances = sorted(regressor.feature_importances_, reverse=True)

plt.title('Feature importances of encoded movie data')
plt.bar(range(len(sorted_importances[0:200])), sorted_importances[0:200])
#plt.show()
plt.tight_layout()
plt.savefig('FeatureImportance.png')
plt.clf()
importances = {}

count = 0
for feature_importance in regressor.feature_importances_:
    if feature_importance > 0.002:
        feature_name = X_names[count]
        importances[feature_name] = feature_importance
    count+=1
    
import operator
sorted_importances = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)
sorted_importances = sorted_importances[0:20]


xs, ys = [*zip(*sorted_importances)]

plt.rcdefaults()
fig, ax = plt.subplots()
plt.barh(xs, ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Top 20 movie revenue predictors')
plt.xlabel('Feature importance score')
#plt.show()
plt.tight_layout()
plt.savefig('RevenuePredictors20.png')
'''
print(importances)
print(type(importances))
print(sorted_importances)
'''

