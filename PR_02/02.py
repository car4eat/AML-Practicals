# apply Filter methods for Feature selection

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv("dataset.csv")

data["Age"].fillna(data["Age"].mean(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# encoding
encoder = LabelEncoder()

data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])

# input and output
x = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = data["Survived"]

print("\n1. Variance Threshold")
selector1 = VarianceThreshold()
selector1.fit(x)

for col, value in zip(x.columns, selector1.variances_):
    print(col, ":", value)

print("\n2. Chi Square")
selector2 = SelectKBest(score_func=chi2, k=5)
selector2.fit(x, y)

for col, score in zip(x.columns, selector2.scores_):
    print(col, ":", score)

print("\n3. ANOVA")
selector3 = SelectKBest(score_func=f_classif, k=5)
selector3.fit(x, y)

for col, score in zip(x.columns, selector3.scores_):
    print(col, ":", score)

print("\n4. Mutual Information")
scores = mutual_info_classif(x, y)

for col, score in zip(x.columns, scores):
    print(col, ":", score)

print("\n5. Correlation Method")
correlation = data.corr(numeric_only=True)

print(correlation["Survived"].sort_values(ascending=False))