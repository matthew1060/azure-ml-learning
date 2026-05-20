import argparse
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output_train", type=str)
parser.add_argument("--output_test", type=str)
args = parser.parse_args()

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs(args.output_train, exist_ok=True)
os.makedirs(args.output_test, exist_ok=True)
train_df.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
test_df.to_csv(os.path.join(args.output_test, "test.csv"), index=False)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")