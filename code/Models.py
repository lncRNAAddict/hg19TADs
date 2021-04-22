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
from sklearn.svm import SVC


def getkmers(file):
    import pandas as pd
    import numpy as np
    #print(file)
    x = pd.read_csv(file,delimiter='\t',header=None)
    x.index = x[0]
    x=x.drop([0],axis=1)
    x=x[1].str.split(',',expand=True)
    x=x.apply(pd.to_numeric,errors='coerce')
    y=x.iloc[:,-1]
    x=x.iloc[:,:-1]
    print(x.shape)
    print(y.shape)
    return (x,y)
  
    
def getRepeats(file,num_bins=10):
    import pandas as pd
    import numpy as np
    x = pd.read_csv(file,delimiter='\t',header=None)
    x.index = x[0]
    x=x.drop([0],axis=1)
    x=x[1].str.split(',',expand=True)
    x=x.apply(pd.to_numeric,errors='coerce')
    y=x.iloc[:,-1]
    x=x.iloc[:,:-1]
    step = int(100/num_bins)
    X = x.iloc[:,0:step].sum(axis=1)
    for i in range(1,num_bins):
        xx=x.iloc[:,i*step:i*step + step].sum(axis=1,numeric_only=True)
        X=pd.concat([X,xx],axis=1)
    print(X.shape)
    return (X,y)
    
def pandas2numpy(xx,num_bins=10):
    import pandas as pd
    import numpy as np
    #remove chrx chry
    #xx = xx[~xx[0].str.contains("chrX")]
    #xx = xx[~xx[0].str.contains("chrY")]
    X1=pd.DataFrame(columns=['Chr','start','end'])
    X1[['Chr','start','end']]=xx[0].str.split(':',expand=True)
    X2=xx[1].strx.apply(label_fold, axis=1)
    y = pd.DataFrame(xx.iloc[:,103])
    y=y.apply(pd.to_numeric,errors='coerce')
    x=xx.drop(['Chr','start','end'],axis=1)
    x=x.apply(pd.to_numeric,errors='coerce').split(',',expand=True)
    xx = pd.concat([X1,X2],axis = 1)
    #xx['fold'] = x
    return x

def pandas2TADLactuca(file,num_bins=10):
    import pandas as pd
    import numpy as np

    x = pd.read_csv(file,delimiter='\t',header=None)
    x.index = x[0]
    x=x.drop([0],axis=1)
    #print(x)
    x=x[1].str.split(',',expand=True)
    #print(x)
    x=x.apply(pd.to_numeric,errors='coerce')
    y=x.iloc[:,-1]
    #print("y")
    #print(x)
    #print("y")
    x=x.iloc[:,:-1]
    #print(x.shape)
    #print(y.shape)
    step = int(100/num_bins)
    X = x.iloc[:,0:step].sum(axis=1)
    for i in range(1,num_bins):
        xx=x.iloc[:,i*step:i*step + step]
        xx = xx.sum(axis=1)
        #print(i)
        #print(xx)
        X=pd.concat([X,xx],axis=1)
    #X=X.sum(axis=1,numeric_only=True)
    #print(X)
    return (X,y)

def getHistones(folder,num_samples,num_features,num_bins=10):
    from os import listdir
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
    from sklearn.model_selection import StratifiedKFold, KFold
    
    files=listdir(folder)
    files=[i for i in files if 'csv' in i]
    files.sort()
    files
    #x = np.empty(shape=(num_samples,num_features+1,len(files)))
    print(files[0])
    (x,y)=pandas2TADLactuca(folder+files[0],num_bins)
    print(x.shape)
    #print(y[0])     
    for i in range(1,len(files)): 
        print(files[i])
        (xx,y)=pandas2TADLactuca(folder+files[i],num_bins)
        #print(y[0])
        print(xx.shape)
        x=pd.concat([x,xx],axis=1)
    if num_bins == 1:
       x=np.reshape(x,(x.shape[0],1))    
    print(x.shape)
    print(y.shape)        
    return (x,y)
 
    
def read_data(file):
    f=h5py.File(file,'r')
    x_train = np.array(f['x_train'])
    x_test = np.array(f['x_test'])
    y_train = np.array(f['y_train'])
    y_test = np.array(f['y_test'])
    f.close()
    return(x_train,x_test,y_train,y_test)

def tuneModel(x_train,x_test,y_train,y_test,classifier,params):    
    gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=params,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=20,verbose=0)
    gd_sr.fit(x_train, y_train)
    model =gd_sr.best_estimator_
    return model

def Prediction(x_train,x_test,y_train,y_test,loaded_model):
    y_pred = loaded_model.predict(x_test)
    print(y_pred)
    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    probs = loaded_model.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(auc)
    print(confusion_matrix(y_test,y_pred))
    

def FeatureImportance(x_test,y_test,model,index_s,index_e):
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    np.random.shuffle(x_test[:,index_s:index_e])
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    auc1 = roc_auc_score(y_test, probs)   
    diff = auc - auc1
    return diff 

import h5py
def savedata(x_train,x_test,y_train,y_test,outfile):
    hf = h5py.File(outfile, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('y_test', data=y_test)
    hf.close()

def readdata(h5file):
    f1 = h5py.File(h5file,'r')    
    x_train = np.array(f1.get('x_train'))
    x_test = np.array(f1.get('x_test'))
    y_train = np.array(f1.get('y_train'))
    y_test = np.array(f1.get('y_test'))
    return(x_train,x_test,y_train,y_test)
    









