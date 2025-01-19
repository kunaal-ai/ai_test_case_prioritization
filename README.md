# ai_test_case_prioritization
Using ML to analyze test cases to prioritize, saves so much manual hard work.


# Visualization
```pip install matplotlib seaborn```

# Test Case Prioritization with Machine Learning

## Overview
This project uses a Random Forest machine learning model to predict and prioritize high-risk test cases based on test case execution history, defects found, and code coverage metrics. The model aims to help identify the most critical test cases for efficient and effective testing.

## Features
- Collects and preprocesses test case data
- Trains a Random Forest model to predict high-risk test cases
- Evaluates model performance with precision, recall, and accuracy metrics
- Provides visualization of predicted results
- Supports automated model retraining with new data

## Dataset
The dataset includes the following columns:
- `test_case_id`: Unique identifier for each test case
- `execution_result`: The result of the test case (pass/fail)
- `defect_severity`: Severity of the defect (low, medium, high)
- `code_coverage`: Percentage of the code covered by the test case

Sample data format:
```plaintext
test_case_id,execution_result,defect_severity,code_coverage
TC001,pass,high,85
TC002,fail,medium,60
TC003,pass,low,75
```

## Installation
Clone the repository:

`git clone https://github.com/your_username/your_repo_name.git
cd your_repo_name`

## Install the required dependencies:
```pip install -r requirements.txt```

# Usage
**Train the Model**: ```sh python train_model.py```

This script will:
- Load and preprocess the dataset
- Train the Random Forest model
- Evaluate the model's performance
- Save the trained model to ```test_case_priority_model.pkl```



**Predict New Test Cases:** ```sh python predict_test_cases.py```

This script will:
- Load the trained model
- Load new test case data
- Preprocess the data and make predictions
- Save the predictions to ```predicted_test_cases.csv```

# Future Enhancements
- Integrate the model into a CI/CD pipeline using GitLab CI/CD
- Implement active learning and incremental model updates
- Explore more complex models like Gradient Boosting Machines (GBM) or Neural Networks