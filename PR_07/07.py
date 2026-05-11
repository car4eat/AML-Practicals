# develop SVM model on loan approval....

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

data = pd.read_csv("loan.csv")

data.fillna(data.mode().iloc[0], inplace=True)

encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = encoder.fit_transform(data[column])

x = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Feature Scaling
scaler = StandardScaler()

x = scaler.fit_transform(x)

# Dataset Visualization
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title("Loan Dataset")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Kernel": SVC(kernel="linear", C=1),
    "RBF Kernel": SVC(kernel="rbf", C=1, gamma=0.1),
    "Polynomial Kernel": SVC(kernel="poly", C=1, degree=3)
}

for name, model in models.items():

    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    print("\n", name)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))

    print("Accuracy:")
    print(accuracy_score(y_test, prediction))

    print("Precision:")
    print(precision_score(y_test, prediction))

    print("Recall:")
    print(recall_score(y_test, prediction))

    print("F1 Score:")
    print(f1_score(y_test, prediction))