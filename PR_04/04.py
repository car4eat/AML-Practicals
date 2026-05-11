# PCA and LDA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

print("\nPCA Reduced Features:")
print(x_pca[:5])

# LDA
lda = LinearDiscriminantAnalysis(n_components=1)
x_lda = lda.fit_transform(x, y)

print("\nLDA Reduced Features:")
print(x_lda[:5])

# train test split for PCA
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x_pca, y, test_size=0.2, random_state=42
)

# model for PCA
model1 = LogisticRegression()

model1.fit(x_train1, y_train1)

prediction1 = model1.predict(x_test1)

accuracy1 = accuracy_score(y_test1, prediction1)

print("\nPCA Accuracy:")
print(accuracy1)

# train test split for LDA
x_train2, x_test2, y_train2, y_test2 = train_test_split(
    x_lda, y, test_size=0.2, random_state=42
)

# model for LDA
model2 = LogisticRegression()

model2.fit(x_train2, y_train2)

prediction2 = model2.predict(x_test2)

accuracy2 = accuracy_score(y_test2, prediction2)

print("\nLDA Accuracy:")
print(accuracy2)

# PCA Plot
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.title("PCA 2D Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# LDA Plot
plt.scatter(x_lda[:, 0], [0]*len(x_lda), c=y)
plt.title("LDA 2D Projection")
plt.xlabel("LDA 1")
plt.show()
