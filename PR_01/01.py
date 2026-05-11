# Apply data preprocessing techniques to make data suitable for machine learning.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset.csv")

print("Dataset Head:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# missing values
print("\nMissing Values:")
print(data.isnull().sum())

# fill missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# outlier detection using boxplot
plt.boxplot(data["Age"])
plt.title("Age Outliers")
plt.show()

q1 = data["Age"].quantile(0.25)
q3 = data["Age"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

data = data[(data["Age"] >= lower) & (data["Age"] <= upper)]

# encoding
encoder = LabelEncoder()

data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])

# before scaling
print("\nBefore Scaling:")
print(data[["Age", "Fare"]].head())

# feature scaling
scaler = StandardScaler()

data[["Age", "Fare"]] = scaler.fit_transform(data[["Age", "Fare"]])

# after scaling
print("\nAfter Scaling:")
print(data[["Age", "Fare"]].head())