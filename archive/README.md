# Lender, a machine learning-based peer-to-peer lending decision support application.

### First work
The `decision tree induction model/loanvisor.py`file contains implementation of the loan prediction model among peer-to-peer lending communities, published [here.](https://www.researchgate.net/profile/Abdul-Gilal/publication/336775046_A_Boosted_Decision_Tree_Model_for_Predicting_Loan_Default_in_P2P_Lending_Communities/links/5db1a569299bf111d4c0a2f8/A-Boosted-Decision-Tree-Model-for-Predicting-Loan-Default-in-P2P-Lending-Communities.pdf)

Supporting explanations, though the code is well-commented, can be found in `decision tree induction model/loanvisorexplanation.txt`.

*Lender* is continously and currently being improved due to the following reasons:
* The need for appropriate cross-validation technique which should account for the class distribution and imbalance.
* Inclusion of feature selection which would correct the presumed misleading information of the recorded accuracies of 98% and 99%.
* Correction of the wrong usage of "Accuracy" as an evaluation metric when there is class imbalance and its distribution in the test data is not accounted for.
* Productivizing the model as a Proof of Concept.

