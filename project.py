import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from IPython.display import display, clear_output
import ipywidgets as widgets


file_path = "/content/data.csv"

if os.path.exists(file_path):
    data = pd.read_csv(file_path, on_bad_lines='skip')
    data.columns = ["password", "strength"]
    print("Dataset Shape:", data.shape)
    display(data.head())
else:
    print("data.csv not found in /content/")
    print("Please upload the dataset file and run this cell again.")
    data = pd.DataFrame()

if data.empty:
    print("Warning: data.csv not found or DataFrame is empty. Creating a synthetic dataset.")
    synthetic_data = {
        'password': ['password123', 'MyStrongP@ssw0rd', 'weak', 'Secure!23'],
        'strength': [0, 2, 0, 1]
    }
    data = pd.DataFrame(synthetic_data)
    print("Synthetic dataset created:")
    display(data.head())


def extract_features(password):
    return [
        len(password),
        sum(c.isdigit() for c in password),
        sum(c.isupper() for c in password),
        sum(c.islower() for c in password),
        sum(not c.isalnum() for c in password)
    ]


X = np.array([extract_features(str(p)) for p in data["password"]])
y = data["strength"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))


def suggest(password):
    suggestions = []
    if len(password) < 8:
        suggestions.append("Increase length to at least 8 characters.")
    if not any(c.isupper() for c in password):
        suggestions.append("Add uppercase letters.")
    if not any(c.islower() for c in password):
        suggestions.append("Add lowercase letters.")
    if not any(c.isdigit() for c in password):
        suggestions.append("Add numbers.")
    if not any(not c.isalnum() for c in password):
        suggestions.append("Add special characters (!@#$%).")
    return suggestions


password_input = widgets.Password(description="Password:")
output = widgets.Output()

def check_password(change):
    with output:
        clear_output()
        pw = password_input.value
        features = np.array([extract_features(pw)])
        prediction = model.predict(features)[0]
        label = {0:"Weak ❌", 1:"Medium ⚠", 2:"Strong ✅"}[prediction]
        print("Predicted Strength:", label)
        print("\nSuggestions:")
        for s in suggest(pw):
            print("-", s)

password_input.observe(check_password, names='value')
display(password_input, output)
