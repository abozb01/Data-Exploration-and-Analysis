import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("your_data.csv")

# Explore data
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Perform advanced statistics
summary_stats = data.describe(include='all')
print(summary_stats)

# Data Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Distribution plot for numerical features
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    sns.histplot(data[column], kde=True, bins=30, label=column)

plt.legend()
plt.show()

# Box plot for categorical features
plt.figure(figsize=(14, 8))
for column in data.select_dtypes(include=['object']).columns:
    sns.boxplot(x=column, y='target', data=data)
    plt.title(f"Boxplot for {column} vs Target")
    plt.show()
