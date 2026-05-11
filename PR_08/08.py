# Case study: Refer to the Bank Marketing dataset  and Apply your own EDA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv("bank-full.csv", sep=";")

print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

# EDA
print("\nTarget Values:")
print(data["y"].value_counts())

data["y"].value_counts().plot(kind="bar")

plt.title("Term Deposit Subscription")
plt.show()

encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = encoder.fit_transform(data[column])

# Manual Feature Selection
x = data[[
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous"
]]

y = data["y"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ML Model
model = RandomForestClassifier()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))

# Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=x.columns
)

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.show()