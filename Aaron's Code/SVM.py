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




xfromdataset =np.load("xtrain_pcaOutputNoScaleNum20.npy")
y = np.load("ytrain_pcaOutputNoScaleNum20.npy")



binWindowSize = 50*(10**6)

y_ = (y/binWindowSize).astype(int)





##################################################################
def cross_validation_for_C(X, y, cparameter):
        clf = SVC(kernel='rbf', C=cparameter, gamma='auto')
        f1=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        f1sum=f1.sum(axis=0)
        f1average=f1sum/3
        return f1average


f1foreachC=np.zeros(13)
best_f1 = None
best_C = None
j=0
C_list = [0.01, 0.1, 1,10 , 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
for c in C_list:
    currentf1=cross_validation_for_C(xfromdataset,y_,c)
    f1foreachC[j]=currentf1
    j=j+1
    if best_f1 is None or currentf1 > best_f1:
        best_f1 = currentf1
        best_C = c





def cross_validation_for_gamma(X, y, gamma):
        clf = SVC(kernel='rbf', C=1.0, gamma=g)
        f1=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        f1sum=f1.sum(axis=0)
        f1average=f1sum/3
        return f1average


f1foreachgamma=np.zeros(13)
best_f1 = None
best_gamma = None
j=0
gamma_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001,0.01,0.1,1,10,100,1000]
for g in gamma_list:
    currentf1=cross_validation_for_gamma(xfromdataset,y_,g)
    f1foreachgamma[j]=currentf1
    j=j+1
    if best_f1 is None or currentf1 > best_f1:
        best_f1 = currentf1
        best_gamma = g



########################################################################

xtest =np.load("xtest_pcaOutputNoScaleNum20.npy")
y2 = np.load("ytest_pcaOutputNoScaleNum20.npy")


binWindowSize = 50*(10**6)

y2_ = (y2/binWindowSize).astype(int)

    
#c3001 = np.load('C300.npy')
#gamma300 = np.load('Gammas300.npy')

best_C=best_C
best_gamma=best_gamma
clf = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
clf.fit(xfromdataset, y_)
y_pred = clf.predict(xtest)
f=f1_score(y2_, y_pred, average='weighted')
a=accuracy_score(y2_, y_pred)


np.save( 'PCAXvalues_50interval_SVM_Correct',xtest)
np.save( 'PCApredictedvalues_50interval_SVM_Correct',y_pred)
np.save( 'PCAactualvalues_50interval_SVM_Correct',y2_)






