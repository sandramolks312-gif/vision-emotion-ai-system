import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv(r"C:\PRACTICE DS PROJECT\AI PROJECTS\MAIN PROJECT\AllDetails\MAIN.csv")
print(df)

x = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values
print("Input:", x)
print("Output:", y)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
print("Label classes:", labelencoder.classes_)
# ['High Focus', 'Low Focus', 'Normal', 'Stress']
# encoded as   [0,           1,          2,        3]
print("Encoded values:", np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y_train))
print(f"Number of classes in training: {num_classes}")

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

sample = np.array([[3, 0, 0.9, 3]])
y_pred1 = classifier.predict(sample)
print("Predicted class:", labelencoder.inverse_transform(y_pred1))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % (accuracy * 100))

# ── Save BOTH files — this is what was missing ──
joblib.dump(classifier,   "fusion_model.pkl")
joblib.dump(labelencoder, "label_encoder.pkl")   # <-- THIS was not being saved before
print("fusion_model.pkl  saved!")
print("label_encoder.pkl saved!")
print("ML Model trained successfully!")