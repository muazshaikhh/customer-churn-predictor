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

scaler = StandardScaler() 

num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges"] # columns with big scaling nums
df[num_cols] = scaler.fit_transform(df[num_cols]) # calcs the average of the data

from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1) # drops churn column so AI can learn on other features
y = df["Churn"] # this is the churn column, essentially the target for the AI

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) # shuffles the data into four, and makes sure the training and testing set have the same amount of churners

print(f"Data sis plit! Training on {len(X_train)} customers, testing on {len(X_test)}.") 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees all voting if customer churns or not, majority wins
model.fit(X_train, y_train) # model is given data and its answers and learns (the real machine learning)

print("Training is complete! Model is done studying the patterns")

from sklearn.metrics import classification_report, accuracy_score

predictions = model.predict(X_test) # model is given only questions now

print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%") # how much it got right
print("\nDetailed Report::")
print(classification_report(y_test, predictions)) # gives us precision, recall, f1-score (balance between precision and recall)

importances = model.feature_importances_ # after the training, random forest remembers which columns helped it make the best decisions and gives them scores
feature_names = X.columns 

feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}) # scores are organized into a table
print("\n-----Top 5 reasons customers may be leaving-----")
print(feature_importance_df.sort_values(by="Importance",ascending=False).head(5)) # puts the main scores at the top








