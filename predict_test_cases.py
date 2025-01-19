import pandas as pd
import pickle

# Load the model
with open('test_case_priority_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load new test case data
#new_test_cases = pd.read_excel('new_test_cases.xlsx')
new_test_cases = pd.read_csv("new_test_cases.csv")


# Preprocess the data for prediction
new_test_cases['defect_severity'] = new_test_cases['defect_severity'].map({'low': 1, 'medium': 2, 'high': 3})
new_test_cases['execution_result'] = new_test_cases['execution_result'].map({'pass': 0, 'fail': 1})

# Predict high-risk test cases
predictions = loaded_model.predict(new_test_cases[['defect_severity', 'code_coverage', 'execution_result']])

# Save the predictions
new_test_cases['priority'] = predictions
new_test_cases.to_csv('predicted_test_cases.csv', index=False)
print("Prediction successful")
