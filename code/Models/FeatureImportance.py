# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:01:45 2020

@author: soibamb
"""


import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def read_data(file):
    #f=h5py.File('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\dixon_h1_histones_feature_sum_data.h5','r')
    f=h5py.File(file,'r')
    x_train = np.array(f['x_train'])
    x_test = np.array(f['x_test'])
    y_train = np.array(f['y_train'])
    y_test = np.array(f['y_test'])
    f.close()
    return(x_train,x_test,y_train,y_test)
    
def tuneRF(x_train,x_test,y_train,y_test,outfile):
    x_train[np.isinf(x_train)] = 0
    x_test[np.isinf(x_test)] = 0
    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)
    grid_param = {
    'n_estimators': [50, 100,200],
    'max_features': ['auto', 'sqrt'],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]}
    gd_sr = GridSearchCV(estimator=RandomForestClassifier(),
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1,verbose=2)
    gd_sr.fit(x_train, y_train)
    from sklearn.externals import joblib
    joblib.dump(gd_sr.best_estimator_, outfile)
    import shap
    model =gd_sr.best_estimator_
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    columns = ['CTCF','DNase','H2A','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K79me2','H3K9ac','H3K9me3','H4K20me1','TAD']
    shap.summary_plot(shap_values[1], x_test,feature_names=columns)

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\gm12878\\50kb\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\gm12878\\50kb\\gm12878.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\50kb\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\h1\\50kb\\hela.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hela\\50kb\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hela\\50kb\\hela.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hmec\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\hmec\\hmec.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\huvec\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\huvec\\huvec.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\imr90\\50kb\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\imr90\\50kb\\imr90.pkl')

(x_train,x_test,y_train,y_test) = read_data('C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\nhek\\feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'C:\\Users\\soibamb\\Dropbox\\Research\\TADs\\boundaries\\histones\\nhek\\nhek.pkl')

