import numpy as np
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    #if not title:
    #    if normalize:
    #        title = 'Normalized confusion matrix'
    #    else:
    #        title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm=np.nan_to_num(cm)
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right") #rotation=45
             #rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

#########################
######     SVM     ######
#########################
ytest_actual=np.load("ytest_actual.npy")
ytest_predicted=np.load("ytest_predicted.npy")
class_names=unique_labels(ytest_actual, ytest_predicted)
print(class_names)
print(ytest_actual)
print(ytest_predicted)
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=False,title='Confusion matrix - SVM (300 intervals)')
plt.savefig('SVM_300_ConfusionMat.png')
plt.close()
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=True,title='Normalized Confusion matrix - SVM (300 intervals)')
plt.savefig('SVM_300_Norm_ConfusionMat.png')
plt.close()
#plt.show()

ytest_actual=np.load("ytest_actual_svm100.npy")
ytest_predicted=np.load("ytest_predicted_svm100.npy")
class_names=unique_labels(ytest_actual, ytest_predicted)
print(class_names)
print(ytest_actual)
print(ytest_predicted)
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=False,title='Confusion matrix - SVM (100 intervals)')
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=True,title='Normalized Confusion matrix - SVM (100 intervals)')
#plt.savefig('SVM_100_ConfusionMat.png')
plt.show()
#plt.close()


#########################
######      RF     ######
#########################
ytest_actual=np.load("ytest_actual_rf.npy")
ytest_predicted=np.load("ytest_pred_rf.npy")
class_names=np.unique(ytest_actual)
print(class_names)
print(ytest_actual)
print(ytest_predicted)
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=False,title='Confusion matrix - Random Forest (100 intervals)')
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=True,title='Normalized Confusion matrix - Random Forest (100 intervals)')
#plt.savefig("RF_100_ConfusionMat.png")
#plt.close()
plt.show()

ytest_actual=np.load("ytest_actual_rf300.npy")
ytest_predicted=np.load("ytest_pred_rf300.npy")
class_names=np.unique(ytest_actual)
print(class_names)
print(ytest_actual)
print(ytest_predicted)
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=False,title='Confusion matrix - Random Forest (300 intervals)')
plt.savefig("RF_300_ConfusionMat.png")
plt.close()
plot_confusion_matrix(ytest_actual, ytest_predicted, classes=class_names, normalize=True,title='Normalized Confusion matrix - Random Forest (300 intervals)')
plt.savefig("RF_300_Norm_ConfusionMat.png")
plt.close()

