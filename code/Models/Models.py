# -*- coding: utf-8 -*-


import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,recall_score
from sklearn.externals import joblib
import shap
from sklearn.svm import SVC

def read_data(file):
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
    joblib.dump(gd_sr.best_estimator_, outfile)
    model =gd_sr.best_estimator_
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    columns = ['CTCF','DNase','H2A','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K79me2','H3K9ac','H3K9me3','H4K20me1','TAD']
    shap.summary_plot(shap_values[1], x_test,feature_names=columns)

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
    
(x_train,x_test,y_train,y_test) = read_data('gm12878_feature_sum_data.h5')
tuneRF(x_train,x_test,y_train,y_test,'gm12878_rf.pkl')
Prediction('gm12878_feature_sum_data.h5','gm12878_rf.pkl')

