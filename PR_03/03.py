# apply Wrapper methods for Feature selection

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("dataset.csv")

# fill missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# encoding
encoder = LabelEncoder()

data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])

# input and output
x = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# logistic regression model
model = LogisticRegression(max_iter=1000)

# Sequential Forward Selection
sfs = SequentialFeatureSelector(
    model,
    n_features_to_select=4,
    direction="forward"
)

sfs.fit(x_train, y_train)

forward_features = x.columns[sfs.get_support()]

print("\nForward Selected Features:")
print(forward_features)

# train model using forward features
model.fit(x_train[forward_features], y_train)

prediction1 = model.predict(x_test[forward_features])

accuracy1 = accuracy_score(y_test, prediction1)

print("\nForward Selection Accuracy:")
print(accuracy1)

# Sequential Backward Elimination
sbe = SequentialFeatureSelector(
    model,
    n_features_to_select=4,
    direction="backward"
)

sbe.fit(x_train, y_train)

backward_features = x.columns[sbe.get_support()]

print("\nBackward Selected Features:")
print(backward_features)

# train model using backward features
model.fit(x_train[backward_features], y_train)

prediction2 = model.predict(x_test[backward_features])

accuracy2 = accuracy_score(y_test, prediction2)

print("\nBackward Elimination Accuracy:")
print(accuracy2)