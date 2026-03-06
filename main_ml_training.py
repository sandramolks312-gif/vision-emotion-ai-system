import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib


df=pd.read_csv(r"C:\PRACTICE DS PROJECT\AI PROJECTS\MAIN PROJECT\AllDetails\MAIN.csv")
print(df)


# Prepare features and target
x = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values
print("Input:", x)
print("Output:", y)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
print("Label classes:", labelencoder.classes_)
print("Encoded values:", np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Get number of unique classes in training data
num_classes = len(np.unique(y_train))
print(f"Number of classes in training: {num_classes}")

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

x = np.array([[3, 0, 0.9, 3]])
y_pred1 = classifier.predict(x)
print("Predicted class:", labelencoder.inverse_transform(y_pred1))

joblib.dump(classifier, "fusion_model.pkl")
# print("ML Model trained successfully!")

# from sklearn import metrics
# print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
# print("Root Mean Square Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
# # accuracy score
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:%.2f" % (accuracy*100))

joblib.dump(classifier, "fusion_model.pkl")
print("ML Model trained successfully!")
