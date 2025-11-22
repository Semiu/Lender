# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 09:07:16 2022

@author: Semiu
"""
#Import required libraries 
import pandas as pd
from sklearn import  metrics
import xgboost as xgb
import joblib

#load the training dataset with the stratified K-folds
train_datasets = pd.read_csv(r"F:\Dataset\Lendingclub\stratifiedLCloandata.csv", low_memory = False)

#All columns except 'is_safe' and 'stratkfold' as training features
train_features = [tf for tf in train_datasets.columns if tf not in ('is_safe', 'stratkfold', 'policy_code')]

#Function for Logistic Regression Classification
def run_xgboost(fold):
    #Get training and validation data using folds
    train_datasets_train = train_datasets[train_datasets.stratkfold != fold].reset_index(drop=True)
    train_datasets_valid = train_datasets[train_datasets.stratkfold == fold].reset_index(drop=True)
    
    #Get train data
    X_train = train_datasets_train[train_features].values
    
    #Get validation data
    X_valid = train_datasets_valid[train_features].values
    
    #Initialize XGboost model
    xgb_model = xgb.XGBClassifier(n_jobs=-1)
    
    #xgb_model = XGBClassifier()
    
    #Fit the model on training data
    xgb_model.fit(X_train, train_datasets_train.is_safe.values)
    
    #Predict on validation
    valid_preds = xgb_model.predict(X_valid)
    
    #Get the ROC AUC score
    auc = metrics.roc_auc_score(train_datasets_valid.is_safe.values, valid_preds)
    
    #save the model
    joblib.dump(xgb_model, f"../ml-models/xgb_model{fold}.bin")
  
    return auc

#Function to calculate the mean of all the auc generated for each of the folds
def calc_mean_auc():
    list_auc = []
    #the number of folds created is 10
    for i in range(10):
        list_auc.append(run_xgboost(i))
    return sum(list_auc)/len(list_auc)
    

if __name__ == "__main__":
    print(calc_mean_auc())