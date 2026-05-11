# pune House Price Prediction Using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# load dataset
data = pd.read_csv("HousingData.csv")

# first 5 rows
print("\nDataset Head:")
print(data.head())

# missing values
print("\nMissing Values:")
print(data.isnull().sum())

# fill missing values
data.fillna(data.mean(), inplace=True)

# boxplot for outliers
data.boxplot(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Outliers")
plt.show()

# remove outliers using IQR
for column in data.columns:

    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    data = data[
        (data[column] >= lower) &
        (data[column] <= upper)
    ]

# input and output
x = data.drop("MEDV", axis=1)
y = data["MEDV"]

# feature scaling
scaler = StandardScaler()

x = scaler.fit_transform(x)

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# training and testing
for name, model in models.items():

    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    print("\n", name)

    print("MAE :", mae)
    print("MSE :", mse)
    print("R2 Score :", r2)

# best model
best_model = RandomForestRegressor()

best_model.fit(x_train, y_train)

# feature importance
importance = pd.Series(
    best_model.feature_importances_,
    index=data.columns[:-1]
)

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# feature importance graph
importance.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.show()