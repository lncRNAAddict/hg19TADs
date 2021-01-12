import numpy as np
import pandas as pd 

import h5py

f=h5py.File('hmec_histones_feature_sum_data.h5','r')
X_train = np.array(f['x_train'])
X_test = np.array(f['x_test'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])
f.close()




from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50,random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("------------------------------------------")




from sklearn.model_selection import cross_val_score
train_score = cross_val_score(classifier, X_train, y_train, scoring = 'accuracy', cv=10)
test_score = cross_val_score(classifier, X_test, y_test, scoring = 'accuracy', cv=10)

print("Train: ", train_score)
print("Mean: ", train_score.mean())
print("STD: ", train_score.std())
print("------------------------------------------")
print("Test: ", test_score)
print("Mean: ", test_score.mean())
print("STD: ", test_score.std())
print("------------------------------------------")


from sklearn.model_selection import GridSearchCV
grid_param = {
    'n_estimators': [50, 100,200],
    'max_features': ['auto', 'sqrt'],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

gd_sr = GridSearchCV(estimator=classifier,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_
print("Best Value: ", best_result)

from sklearn.metrics import roc_auc_score
probs = gd_sr.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


print('----------------------------------------')
# save the model to disk
import pickle 
filename = 'Randomforest.pickle'
filename = 'RandomFR-Tuning.pickle'
pickle.dump(classifier, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Test Score: " ,result)
