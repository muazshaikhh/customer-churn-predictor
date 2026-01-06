import pandas as pd

df = pd.read_csv("customer_churn_data.csv")
print(df.head()) # outputs first 10 lines

print(df.isnull().sum()) # outputs amount of null answers per each
print(df.info()) # tells us which values are float or object, which we need to fix

df = df.drop("CustomerID", axis=1) # irrelevant data
df["InternetService"] = df["InternetService"].fillna("None") # replace void answers
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}) # change values to num

df = pd.get_dummies(df, columns=["Gender", "ContractType", "InternetService", "TechSupport"], drop_first = True) # splits columns into simpler ones and leaves out redundant trues
print(df.columns)

from sklearn.preprocessing import StandardScaler




