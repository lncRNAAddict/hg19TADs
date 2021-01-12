# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:02:08 2020

@author: soibamb
"""
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,recall_score

def read_data(file):
    #f=h5py.File('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\dixon_h1_histones_feature_sum_data.h5','r')
    f=h5py.File(file,'r')
    x_train = np.array(f['x_train'])
    x_test = np.array(f['x_test'])
    y_train = np.array(f['y_train'])
    y_test = np.array(f['y_test'])
    f.close()
    return(x_train,x_test,y_train,y_test)

def Prediction(datafile,filename):
    (x_train,x_test,y_train,y_test) = read_data(datafile)
    x_train[np.isinf(x_train)] = 0
    x_test[np.isinf(x_test)] = 0
    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)
    loaded_model = joblib.load(filename)
    y_pred = loaded_model.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    probs = loaded_model.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(auc)


Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\gm12878\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\gm12878\\gm12878.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\h1.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hela\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hela\\hela.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hmec\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hmec\\hmec.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\huvec\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\huvec\\huvec.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\imr90\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\imr90\\imr90.pkl')
Prediction('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\nhek\\feature_sum_data.h5','C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\nhek\\nhek.pkl')