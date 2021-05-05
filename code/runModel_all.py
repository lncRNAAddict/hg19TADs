#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import os
import pandas as pd
import numpy as np



# In[452]:
import Models
import imp
imp.reload(Models)

import os,sys
folder = sys.argv[1] 
num_bins_histone = int(sys.argv[2])
num_bins_sequence = int(sys.argv[3])

x,y = Models.getHistones(folder,num_bins_histone)

num_bins=1
x2,y = Models.getkmers(sys.argv[4]) #kmer
x3,y = Models.getRepeats(sys.argv[6],num_bins_sequence) # conservation
x4,y = Models.getRepeats(sys.argv[7],num_bins_sequence) # LINE
x5,y = Models.getRepeats(sys.argv[8],num_bins_sequence) #SINE
x6,y = Models.getRepeats(sys.argv[9],num_bins_sequence) #LTR
x7,y = Models.getRepeats(sys.argv[10],num_bins_sequence) #DNA
x8,y = Models.getkmers(sys.argv[5]) #promoter


x = pd.concat([x,x2,x3,x4,x5,x6,x7,x8],axis=1)
print(x.shape)

# In[456]:

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
feature_scaler = StandardScaler()
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



grid_param = {
    'n_estimators': [100,200,500,1000],
    'max_features': ['auto','sqrt','log2'],
    'criterion': ['gini','entropy'],
    'bootstrap': [True]}


from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=42)
x_train1 = feature_scaler.fit_transform(x_train)
x_test1 = feature_scaler.transform(x_test)

gd_sr1 = Models.tuneModel(x_train1,x_test1,y_train,y_test,RandomForestClassifier(),grid_param)
Models.Prediction(x_train1,x_test1,y_train,y_test,gd_sr1)

'''
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,70,70+64))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,134,135))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,135,136))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,136,137))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,137,138))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,138,139))
print(Models.FeatureImportance(x_test1,y_test,gd_sr1,139,140))
print("histones")
for i in range(0,14):
    print(Models.FeatureImportance(x_test1,y_test,gd_sr1,i*5,i*5+5))
'''




