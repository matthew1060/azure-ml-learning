import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_test", type=str)
parser.add_argument("--input_model", type=str)
args = parser.parse_args()

test_df = pd.read_csv(os.path.join(args.input_test, "test.csv"))
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]
model = joblib.load(os.path.join(args.input_model, "model.pkl"))

mlflow.start_run()
accuracy = accuracy_score(y_test, model.predict(X_test))
mlflow.log_metric("test_accuracy", accuracy)
mlflow.end_run()
print(f"Test accuracy: {accuracy}")