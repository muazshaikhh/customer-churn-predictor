# Customer Churn Predictor

A machine learning pipeline using a Random Forest Classifier to predict the probability of customer churn based on usage and demographic data.

## Performance
- **Accuracy:** 99.5%
- **Model:** Random Forest (100 Decision Trees)
- **Top Predictors:** Tech Support, Contract Type, and Monthly Charges.

## Setup
1. Install dependencies:
   ```pip install pandas scikit-learn```

2. Run the application:
   ```python predict_new_customer.py```

## Dependencies
- pandas
- scikit-learn
- joblib

## Description
- Uses a pre-trained Random Forest model to predict customer churn
- Preprocesses data, handles missing values, encodes categorical variables, and scales numerical features
- The trained model and scaler are already saved in the models/ folder
- ```predict_new_customer.py``` can be run directly to predict churn risk for new customers, its values can be edited