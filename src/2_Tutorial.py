
import os

from azureml.core import Run
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Get the experiment run context
run = Run.get_context()

# Prepare the dataset.
data = pd.read_csv("./data/diabetes.csv")

# Split data.
X, y = data.iloc[:, :-1], data.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Logistic regression.
reg = 0.1
model = LogisticRegression(C = 1/reg, solver = "liblinear").fit(X_train, y_train)

# Calculate metrics.
y_hat = model.predict(X_test)
run.log("Accuracy", accuracy_score(y_test, y_hat))

# Save the trained models.
os.makedirs("outputs", exist_ok = True)
joblib.dump(value = model, filename = "outputs/model.pkl")

run.complete()