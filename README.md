# Customer Churn Predictor

A machine learning project using a Random Forest Classifier to predict the probability of customer churn.

## Performance
- **Accuracy:** 99.5%
- **Model:** Random Forest (100 Decision Trees)

## Setup
1. Install dependencies:
   ```pip install pandas scikit-learn joblib```

2. Run the application:
   ```python predict_new_customer.py```

## Description
- Uses a pre-trained Random Forest model to predict customer churn
- Handles missing values, encodes categorical variables, and scales numerical features
- Model and scaler are saved in the ```models/``` folder
- Input values in ```predict_new_customer.py``` can be edited to test new customers