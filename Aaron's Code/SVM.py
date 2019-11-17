import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.datasets import  load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



###################################

#########################################
xfromXGB =np.load("xtrain_xgFeatures_156.npy")
y = np.load("ytrain_xgFeatures_156.npy")

binWindowSize = 300*(10**6)

y_ = (y/binWindowSize).astype(int)

##################################################################

def cross_validation_for_C(X, y, cparameter):
        clf = SVC(kernel='rbf', C=cparameter, gamma='auto')
        auc=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        aucsum=auc.sum(axis=0)
        aucaverage=aucsum/3
        return aucaverage


aucforeachC=np.zeros(13)
best_auc = None
best_C = None
j=0
C_list = [0.01, 0.1, 1,10 , 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
for c in C_list:
    currentauc=cross_validation_for_C(xfromXGB,y_,c)
    aucforeachC[j]=currentauc
    j=j+1
    if best_auc is None or currentauc > best_auc:
        best_auc = currentauc
        best_C = c





def cross_validation_for_gamma(X, y, gamma):
        clf = SVC(kernel='rbf', C=1.0, gamma=g)
        auc=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        aucsum=auc.sum(axis=0)
        aucaverage=aucsum/3
        return aucaverage


aucforeachgamma=np.zeros(13)
best_auc = None
best_gamma = None
j=0
gamma_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001,0.01,0.1,1,10,100,1000]
for g in gamma_list:
    currentauc=cross_validation_for_gamma(xfromXGB,y_,g)
    aucforeachgamma[j]=currentauc
    j=j+1
    if best_auc is None or currentauc > best_auc:
        best_auc = currentauc
        best_gamma = g

  
########################################################################

xtest =np.load("xtest_xgFeatures_156.npy")
y2 = np.load("ytest_xgFeatures_156.npy")


binWindowSize = 300*(10**6)

y2_ = (y2/binWindowSize).astype(int)

    
#c3001 = np.load('C300.npy')
#gamma300 = np.load('Gammas300.npy')

best_C=100
best_gamma=1e-7
clf = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
clf.fit(xfromXGB, y_)
y_pred = clf.predict(xtest)
f=f1_score(y2_, y_pred, average='weighted')
a=accuracy_score(y2_, y_pred)


np.save( 'Xvalues',xtest)
np.save( 'predictedvalues',y_pred)
np.save( 'actualvalues',y2_)






