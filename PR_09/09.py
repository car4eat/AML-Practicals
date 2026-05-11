# Build a Decision Tree model for a below given dataset and evaluate its performance using
# accuracy, precision, recall, and F1-score. Interpret the results.

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

data = pd.read_csv("tennis.csv")

print(data.head())

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

x = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

print("\nPrecision:")
print(precision_score(y_test, prediction))

print("\nRecall:")
print(recall_score(y_test, prediction))

print("\nF1 Score:")
print(f1_score(y_test, prediction))

# Decision Tree Visualization
plt.figure(figsize=(10,6))

plot_tree(
    model,
    feature_names=x.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.show()

print("\nInterpretation:")
print("Higher accuracy and F1-score indicate better model performance.")