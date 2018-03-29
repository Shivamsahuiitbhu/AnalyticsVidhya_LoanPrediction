# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 19:58:58 2018

@author: Lenovo
"""

# prediction by model
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test = train_test_split(df,target_variable,test_size=0.25)


# tunning hyperparameter by rabdomized seachCV
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in range(60,320,20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(3,16,2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in range(40,250,20)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in range(20,120,20)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
    
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)

pprint(rf_random.best_params_)



def evaluate(model,xtest, ytest):
    predictions = model.predict(xtest)
    ROC_Score = metrics.roc_auc_score(ytest,predictions)
    accuracy  = metrics.accuracy_score(predictions,ytest)
    print('Model Performance')
    print('ROC Score: %s' % '{0:.4%}'.format(ROC_Score))
    print("Accuracy : %s" % "{0:.4%}".format(accuracy))
    print('\n')
    
    return accuracy

base_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test,y_test)
print(rf_random.best_score_)


# tunning parameter by GridSearch

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
# Number of trees in random forest
n_estimators = [int(x) for x in range(80,400,20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(3,16,2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in range(40,400,20)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in range(20,200,20)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {'max_features':range(2,8,1)}

# Create a based model
rf = RandomForestClassifier(n_estimators=160,max_depth=7,
                            bootstrap = True,max_features='sqrt',
                            min_samples_leaf=20,min_samples_split= 60)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2,scoring='roc_auc')

# Fit the grid search to the data
grid_search.fit(df, target_variable)
print(grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_)