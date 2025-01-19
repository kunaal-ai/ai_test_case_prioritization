import pandas as pd
import numpy as np

# Set the number of new test cases
num_new_cases = 500

# Generate synthetic data for new test cases
new_data = {
    "test_case_id": ["TC" + str(i).zfill(4) for i in range(1001, 1001 + num_new_cases)],
    "execution_result": np.random.choice(["pass", "fail"], num_new_cases),
    "defect_severity": np.random.choice(["low", "medium", "high"], num_new_cases),
    "code_coverage": np.random.randint(50, 100, num_new_cases),
}

# Create a DataFrame
new_df = pd.DataFrame(new_data)

# Save the new test cases dataset to a CSV file
new_file_path = "new_test_cases.csv"
new_df.to_csv(new_file_path, index=False)

print(
    f"New synthetic dataset with {num_new_cases} test cases created and saved to {new_file_path}"
)
