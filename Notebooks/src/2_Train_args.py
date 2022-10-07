
import os

from azureml.core import Run
import pandas as pd
import joblib
import argparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get the experiment run context
run = Run.get_context()

# Set reg hyperparameter.
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type = float, dest = 'reg_rate', default = 0.01)
args = parser.parse_args()
reg = args.reg_rate

# Prepare dataset.
data = pd.read_csv('./data/diabetes.csv')

# Split data.
X, y = data.iloc[:, :-1], data.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Logistic regression.
model = LogisticRegression(C = 1/reg, solver = "liblinear").fit(X_train, y_train)

# Calculate metrics.
y_hat = model.predict(X_test)
run.log("Accuracy", accuracy_score(y_test, y_hat))
run.log("reg", reg)

# Save the trained models.
os.makedirs("outputs", exist_ok = True)
joblib.dump(value = model, filename = "outputs/model.pkl")

run.complete()