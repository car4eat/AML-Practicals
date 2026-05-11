import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")

X = df.drop("Purchase", axis=1)
y = df["Purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt = DecisionTreeClassifier(random_state=42)

bagging = BaggingClassifier(
    estimator=dt,
    n_estimators=10,
    random_state=42
)

bagging.fit(X_train, y_train)

bagging_pred = bagging.predict(X_test)

print("Bagging Accuracy:", accuracy_score(y_test, bagging_pred))

adaboost = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

adaboost.fit(X_train, y_train)

adaboost_pred = adaboost.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, adaboost_pred))

base_models = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression())
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

stacking.fit(X_train, y_train)

stacking_pred = stacking.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test, stacking_pred))