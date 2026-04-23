## LOAN-APPORVAL-PREDICTOR

1.**Data generation** :   In this Project  it starts by generating a synthetic loan data uisng using **data/generate_data.py** file and also  crates a 5000 fake loan applicant with 21 different records like (Credit_score, loan_amount , employment_years , debt or income_ratio , education , loan purpose and more it will automatically detects whether the loan applicant or applier is approved for the loan or not based on the scoring formula 


2.***Model trianing*** : Loads te generated dataset and trains **4 machines** learning models : Logistic Regression, Random Forest, XGBoost, and a Stacking Ensemble. Before training, the data is preprocessed — numerical features are scaled and categorical features are encoded. The Stacking Ensemble combines all 3 models (XGBoost + Random Forest + Gradient Boosting) and uses Logistic Regression as the final decision maker. After training, it saves the best model and also generates a confusion matrix and ROC curve chart.


3.***Prediction*** : **predictor.py** loads the saved model and takes a loan applicant's details as input. It returns the approval probability, final decision (APPROVED or DENIED), and confidence level (High, Medium or Low). It also provides SHAP explanations showing which features most influenced the decision.


4.***Flask API*** : **api/app.py** wraps the predictor into a REST API with 2 endpoints — /health to check if the server is running and /predict to send applicant data and get a loan decision back in JSON format.


5.***Testing*** : **tests/test_predictor.py** contains unit tests that verify the model loads correctly, predictions return the right format, probability is between 0 
and 1, good applicants get approved and risky applicants get denied.
