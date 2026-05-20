# Updated training script - CI/CD test
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_train", type=str)
parser.add_argument("--output_model", type=str)
args = parser.parse_args()

train_df = pd.read_csv(os.path.join(args.input_train, "train.csv"))
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

mlflow.start_run()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
mlflow.log_param("n_estimators", 100)

os.makedirs(args.output_model, exist_ok=True)
joblib.dump(model, os.path.join(args.output_model, "model.pkl"))
mlflow.end_run()
print("Model trained and saved")