import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
# from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score






xfromdataset =np.load("xtrain_xgFeatures_156.npy")
y = np.load("ytrain_xgFeatures_156.npy")


binWindowSize = 50*(10**6)

y_ = (y/binWindowSize).astype(int)


#pca = PCA(n_components=20)
#x_pca = pca.fit_transform(xfromdataset)


##########################

def cross_validation_for_Depth(X, y, depthparameter):
        clf=RandomForestClassifier(n_estimators=10, max_depth=depthparameter,  random_state=0)
        f1=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        f1sum=f1.sum(axis=0)
        f1average=f1sum/3
        return f1average


f1foreachDepth=np.zeros(13)
best_f1 = None
best_Depth = None
j=0
Depth_list = [1, 10, 20, 30, 40, 50, 70, 80, 100, 120, 150, 200, 250]
print("Depth,"  ",F1")
print("\n")
for d in Depth_list:
    currentf1=cross_validation_for_Depth(xfromdataset,y_,d)
    print(d,"  ",currentf1)
    f1foreachDepth[j]=currentf1
    j=j+1
    if best_f1 is None or currentf1 > best_f1:
        best_f1 = currentf1
        best_Depth = d
print(f1foreachDepth)
#plt.plot(Depth_list, f1foreachDepth)
#plt.xlabel('Depth')
#plt.ylabel('F1 score')
#plt.ylim(0.45,0.6)
#plt.title('Depth vs F1 score')
#plt.show()


def cross_validation_for_est(X, y, est):
        clf=RandomForestClassifier(n_estimators=est, max_depth=None,  random_state=0)
        f1=cross_val_score(clf, X, y, cv=3,  scoring='f1_weighted')
        f1sum=f1.sum(axis=0)
        f1average=f1sum/3
        return f1average


f1foreachest=np.zeros(7)
best_f1 = None
best_est = None
j=0
print("\n")
print("est_list,"  ",F1")
print("\n")
est_list = [1,10, 20, 50, 80, 100, 200]
for e in est_list:
    currentf1=cross_validation_for_est(xfromdataset,y_,e)
    print(e,"  ",currentf1)
    f1foreachest[j]=currentf1
    j=j+1
    if best_f1 is None or currentf1 > best_f1:
        best_f1 = currentf1
        best_est = e

#print(f1foreachest)
#plt.ylim(0.5,0.7)
#plt.plot(est_list, f1foreachest)
#plt.xlabel('n_estimators')
#plt.ylabel('F1 score')
#plt.title(' F1 score vs n_estimators')
#plt.show()

#np.save( 'Estimators100', f1foreachest)
#np.save( 'Depth100', f1foreachDepth)
  
xtest =np.load("xtest_xgFeatures_156.npy")
y2 = np.load("ytest_xgFeatures_156.npy")

binWindowSize = 50*(10**6)

y2_ = (y2/binWindowSize).astype(int)

best_depth=best_Depth
best_Est=best_est
clf=RandomForestClassifier(n_estimators=best_Est, max_depth=best_depth,  random_state=0)
clf.fit(xfromdataset, y_)
y_pred = clf.predict(xtest)
f=f1_score(y2_, y_pred, average='weighted')
a=accuracy_score(y2_, y_pred)
print(f,a)


###########################
