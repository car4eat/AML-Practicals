import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv("emails.csv")

print(data.head())

x = data["text"]
y = data["spam"]

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))

email = ["Congratulations! You won a free iPhone"]

email = vectorizer.transform(email)

result = model.predict(email)

print("\nCustom Email Prediction:")

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")