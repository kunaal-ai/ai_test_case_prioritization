import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predicted test cases dataset
file_path = 'predicted_test_cases.csv'
data = pd.read_csv(file_path)

# Plot settings
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Visualize the distribution of test case priorities
sns.countplot(x='priority', data=data)
plt.title('Distribution of Test Case Priorities')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.show()

# Scatter plot to visualize code coverage vs. priority
plt.figure(figsize=(12, 6))
sns.scatterplot(x='code_coverage', y='priority', hue='defect_severity', data=data)
plt.title('Code Coverage vs. Priority')
plt.xlabel('Code Coverage')
plt.ylabel('Priority')
plt.legend(title='Defect Severity')
plt.show()

# Box plot to visualize the relationship between code coverage and execution result
plt.figure(figsize=(12, 6))
sns.boxplot(x='execution_result', y='code_coverage', data=data)
plt.title('Code Coverage by Execution Result')
plt.xlabel('Execution Result')
plt.ylabel('Code Coverage')
plt.show()

# Heatmap to visualize correlation between features
plt.figure(figsize=(8, 6))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()
