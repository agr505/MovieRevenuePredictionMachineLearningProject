

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

dataset_X_reimported = pd.read_csv('../SanmeshCodes/Encoded_X.csv')
dataset_y_reimported = pd.read_csv('../SanmeshCodes/Encoded_y - revenue.csv') #or - rating.csv
print("Read")

cols = {'5':'Film runtime', '0':'Film budget', '4':'Day of year', '7':'Year of release'}

dataset_X_reimported.rename(columns= cols, inplace=True)
 

dataReImported = pd.concat([dataset_X_reimported,dataset_y_reimported],axis=1)
dataReImported = dataReImported.replace([np.inf, -np.inf], np.nan)
dataReImported = dataReImported.dropna() #just two rows are lost by dropping NaN values. Better than using mean here


X = dataReImported.iloc[:, 1:-2].values
y = dataReImported.iloc[:, -1].values
count = 1
X_names = dataReImported.columns[1:-2].values


regressor = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
                         subsample = 0.8, objective='reg:squarederror')
regressor.fit(X, y)

# Imp Features 
sorted_importances = sorted(regressor.feature_importances_, reverse=True)

plt.title('Feature importances of encoded movie data')
plt.bar(range(len(sorted_importances[0:200])), sorted_importances[0:200])
#plt.show()
plt.tight_layout()
# plt.savefig('FeatureImportance.png')
plt.show()
plt.clf()
importances = {}

count = 0
for feature_importance in regressor.feature_importances_:
	# set threshold manually
    if feature_importance > 0.002:
        feature_name = X_names[count]
        importances[feature_name] = feature_importance
    count+=1
print("Number of vals ", count)
print("")
    
import operator
sorted_importances = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)
sorted_importances = sorted_importances[0:25]


xs, ys = [*zip(*sorted_importances)]

plt.rcdefaults()
fig, ax = plt.subplots()
plt.barh(xs, ys)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Top 25 movie revenue predictors')
plt.xlabel('Feature importance score')
#plt.show()
plt.tight_layout()
# plt.savefig('RevenuePredictors20.png')
plt.show()
'''
print(importances)
print(type(importances))
print(sorted_importances)
'''

