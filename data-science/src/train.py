# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")

    # Step 1: Define input arguments
    parser.add_argument("--train_data", type=str, help="Path to the training dataset")
    parser.add_argument("--test_data", type=str, help="Path to the testing dataset")
    parser.add_argument("--model_output", type=str, help="Output directory to save the trained model")

    # RandomForest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the Random Forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees")

    args = parser.parse_args()
    return args


def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Step 2: Read datasets
    print("Reading datasets...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    # Step 3: Split data into features (X) and target (y)
    target_column = "price"  # Example: target variable for car price prediction
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Step 4: Initialize and train model
    print("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 5: Log hyperparameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Step 7: Log metrics and save model
    mlflow.log_metric("mse", mse)
    print(f"Model Evaluation Complete. MSE: {mse}")

    # Step 8: Save model
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model")
    mlflow.sklearn.save_model(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":

    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

