import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = pd.read_csv("./data/customer_churn_data.csv")
df = df.drop("CustomerID", axis=1) # irrelevant data
df["InternetService"] = df["InternetService"].fillna("None") # replace void answers
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}) # change values to num

# splits columns into simpler ones and leaves out redundant trues
df = pd.get_dummies(df, columns=["Gender", "ContractType", "InternetService", "TechSupport"], drop_first = True) 


scaler = StandardScaler() 
num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"] # columns with big scaling nums
df[num_cols] = scaler.fit_transform(df[num_cols]) # calcs the average of the data


X = df.drop("Churn", axis=1) # drops churn column so AI can learn on other features
y = df["Churn"] # this is the churn column, essentially the target for the AI

# shuffles the data into four, and makes sure the training and testing set have the same amount of churners
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) 

model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees all voting if customer churns or not, majority wins
model.fit(X_train, y_train) # model is given data and its answers and learns (the real machine learning)

# save the trained model so it doesnt retrain everytime
joblib.dump(model, "models/random_forest_churn.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# how much it got right
predictions = model.predict(X_test) # model is given only questions now, so it gets tested
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%") 

# gives us precision, recall, f1-score (balance between precision and recall)
print("\nDetailed Report::")
print(classification_report(y_test, predictions)) 
