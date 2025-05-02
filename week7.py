import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore the structure
print("\nDataset information:")
df.info()

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean the dataset (handling missing values if any)
# In this case, the Iris dataset is clean, but for demonstration:
if df.isnull().sum().any():
    # For numerical columns, we can fill with the mean
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    # For categorical columns, we can fill with the mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("\nMissing values handled.")
else:
    print("\nNo missing values found.")

# Task 2: Basic Data Analysis
print("\nBasic statistics of numerical columns:")
print(df.describe())


# Grouping by species and computing the mean of features
print("\nMean of features grouped by species:")
print(df.groupby('target')[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].mean())

print("\nInteresting findings:")
print("- The average petal length appears to vary significantly between species.")
print("- Setosa generally has smaller petal dimensions compared to versicolor and virginica.")

# Task 3: Data Visualization
plt.figure(figsize=(10, 6))
# Line chart (not directly applicable to this dataset in a meaningful way,
# but demonstrating the syntax for a time series if we had one)
# For demonstration, let's plot sepal length for the first 50 samples
plt.plot(df.index[:50], df['sepal length (cm)'][:50], marker='o', linestyle='-', label='Sepal Length')
plt.title('Sepal Length of First 50 Iris Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
# Bar chart: Average petal length per species
sns.barplot(x='target', y='petal length (cm)', data=df, palette='viridis')
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.show()

plt.figure(figsize=(8, 6))
# Histogram of sepal width
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
# Scatter plot: Sepal length vs. petal length
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='Set2')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', labels=iris.target_names)
plt.grid(True)
plt.show()