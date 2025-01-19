import pandas as pd
import numpy as np

# Set the number of rows for the dataset
num_rows = 10000

# Generate synthetic data
data = {
    'test_case_id': ['TC' + str(i).zfill(3) for i in range(1, num_rows + 1)],
    'execution_result': np.random.choice(['pass', 'fail'], num_rows),
    'defect_severity': np.random.choice(['low', 'medium', 'high'], num_rows),
    'code_coverage': np.random.randint(50, 100, num_rows)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
file_path = 'training_test_case_data_for_priority.csv'
df.to_csv(file_path, index=False)

print(f'Synthetic dataset with {num_rows} rows created and saved to {file_path}')
