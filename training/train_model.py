import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# If using xlsx
#FILE_PATH = "test_cases_priority.xlsx"
#data = pd.read_excel(FILE_PATH)

# if using CSV
data = pd.read_csv("training/training_test_case_data_for_priority.csv")

# Assuming 'defect_severity', 'code_coverage', and '
# execution_result' as features
# We'll need to convert categorical variables to numerical if necessary
# (e.g., defect_severity)
data["defect_severity"] = data["defect_severity"].map(
    {"low": 1, "medium": 2, "high": 3}
)
data["execution_result"] = data["execution_result"].map({"pass": 0, "fail": 1})

X = data[["defect_severity", "code_coverage", "execution_result"]]
y = data["execution_result"]
# Let's assume we are predicting 'execution_result'


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model to a file
with open("model/test_case_priority_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
