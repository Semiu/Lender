# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 19:47:59 2022

@author: Semiu
"""
#Import required libraries 
import pandas as pd
import joblib
from sklearn import linear_model, metrics, preprocessing


#load the training dataset with the stratified K-folds
train_datasets = pd.read_csv(r"F:\Dataset\Lendingclub\stratifiedLCloandata.csv", low_memory = False)

#All columns except 'is_safe' and 'stratkfold' as training features
train_features = [tf for tf in train_datasets.columns if tf not in ('is_safe', 'stratkfold', 'policy_code')]

#Function for Logistic Regression Classification
def run_lr(fold):
  #Get training and validation data using folds
  train_datasets_train = train_datasets[train_datasets.stratkfold != fold].reset_index(drop=True)
  train_datasets_valid = train_datasets[train_datasets.stratkfold == fold].reset_index(drop=True)
  
  #Initialize OneHotEncoder from scikit-learn, and fit it on training and validation features
  ohe = preprocessing.OneHotEncoder()
  full_data = pd.concat(
    [train_datasets_train[train_features], train_datasets_valid[train_features]],
    axis = 0
    )
  ohe.fit(full_data[train_features])
  
  #transform the training and validation data
  x_train = ohe.transform(train_datasets_train[train_features])
  x_valid = ohe.transform(train_datasets_valid[train_features])

  #Initialize the Logistic Regression Model
  lr_model = linear_model.LogisticRegression()

  #Fit model on training data
  lr_model.fit(x_train, train_datasets_train.is_safe.values)

  #Predict on the validation data using the probability for the AUC
  valid_preds = lr_model.predict_proba(x_valid)[:, 1]

  #Get the ROC AUC score
  auc = metrics.roc_auc_score(train_datasets_valid.is_safe.values, valid_preds)
  
  #save the model
  joblib.dump(lr_model, f"../ml-models/lr_model{fold}.bin")

  return auc

#Function to calculate the mean of all the auc generated for each of the folds
def calc_mean_auc():
    list_auc = []
    #the number of folds created is 10
    for i in range(10):
        list_auc.append(run_lr(i))
    return sum(list_auc)/len(list_auc)
    

if __name__ == "__main__":
    print(calc_mean_auc())
        
