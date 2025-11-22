Lender, a machine learning-based peer-to-peer lending decision support application. 

It leverages both the predictive and prescriptive strengths of machine learning models to classify loan applications as safe or unsafe and prescribe what can be done for an unsafe application to be classified safe in future re-application. 

The application would reduce human error and bias in the underwriting process, save time and money, and improve the process’s integrity. 

Safe applications have a higher probability (above a particular threshold) of repayment success, while unsafe applications have a lower probability (below a particular threshold) of repayment success. 

The data used for this project can be downloaded from [here.](http://web.archive.org/web/20140706042617/https://www.lendingclub.com/info/download-data.action)

Currently, the preliminary work done for the proposal and finalist interview stages, presented in this folder, is further arranged in four (4) sub-folders.

`1-data-cleaning-EDA` folder contains notebooks that performed data merging, different data cleaning processes, and the exploratory data analysis (EDA). The `datamerging` notebook contains scripts for the merging of the dataset, while the `datacleansing` notebook contains scripts for the data cleansing. Lastly, the `eda-cv` notebook contains script for the EDA and cross-validation data generation.

`2-basic-model` folder contains Python scripts for the models of the project. A logisitic regression model is developed by the script in `log_regression.py` file, and the XGboost model is developed with the script in `xgboost.py` file.

The Area Under Curve values for the Logistic Regression and XGBoost models are 0.9998477692389542 and 0.9999973618952145, respectively.

`3-features-ranking` folder contains a notebook for the features importance ranking of the loan data using the XGBoost feature importance function.

The models generated for each of folds for both the logistic regression and xgboost models are saved in the `ml-models` folder.
