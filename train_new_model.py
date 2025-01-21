import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create directory for model if it doesn't exist
os.makedirs('model', exist_ok=True)

# Generate synthetic data for training
num_samples = 1000
np.random.seed(42)

# Generate synthetic training data
data = {
    'test_case_id': [f'TC{i:03d}' for i in range(num_samples)],
    'execution_result': np.random.choice(['pass', 'fail'], num_samples),
    'defect_severity': np.random.choice(['low', 'medium', 'high'], num_samples),
    'code_coverage': np.random.randint(50, 100, num_samples)
}

df = pd.DataFrame(data)

# Preprocess the data
df['defect_severity'] = df['defect_severity'].map({'low': 1, 'medium': 2, 'high': 3})
df['execution_result'] = df['execution_result'].map({'pass': 0, 'fail': 1})

# Create target variable (priority) based on rules
df['priority'] = ((df['defect_severity'] >= 2) & 
                 (df['code_coverage'] < 80) & 
                 (df['execution_result'] == 1)).astype(int)

# Features for training
X = df[['defect_severity', 'code_coverage', 'execution_result']]
y = df['priority']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('model/test_case_priority_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

# Save sample test data for demo
sample_data = df[['test_case_id', 'execution_result', 'defect_severity', 'code_coverage']].head(10)
sample_data['execution_result'] = sample_data['execution_result'].map({0: 'pass', 1: 'fail'})
sample_data['defect_severity'] = sample_data['defect_severity'].map({1: 'low', 2: 'medium', 3: 'high'})
sample_data.to_csv('data/sample_test_cases.csv', index=False)
print("Sample test data saved to data/sample_test_cases.csv")
