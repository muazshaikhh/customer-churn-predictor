import pandas as pd
import joblib

# load saved model and scaler
model = joblib.load("models/random_forest_churn.pkl")
scaler = joblib.load("models/scaler.pkl")

# make a new fake customer to personally test model
new_customer = pd.DataFrame({
    'Age': [45],
    'Tenure': [2],
    'MonthlyCharges': [95.00],
    'TotalCharges': [190.00],
    'Gender_Male': [1],
    'ContractType_One-Year': [0],
    'ContractType_Two-Year': [0],
    'InternetService_Fiber Optic': [1],
    'InternetService_None': [0],
    'TechSupport_Yes': [0]
})

# scale new customers data as before
num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]
new_customer[num_cols] = scaler.transform(new_customer[num_cols])

# get prediction
risk_score = model.predict_proba(new_customer)[0][1] # probability of Churn (class 1)
print(f"-----INDIVIDUAL CHURN ASSESSMENT-----")
print(f"Risk of Churn: {risk_score * 100:.1f}%")