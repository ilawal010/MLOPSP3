# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Path to raw data (CSV)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Path to train data output
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Path to test data output
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Ratio for splitting
    args = parser.parse_args()

    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Step 1: Perform label encoding for categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Step 2: Split dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Step 3: Ensure directories exist
    os.makedirs(os.path.dirname(args.train_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_data), exist_ok=True)

    # Step 4: Save datasets
    train_df.to_csv(args.train_data, index=False)
    test_df.to_csv(args.test_data, index=False)

    # Step 5: Log metrics to MLflow
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows", len(test_df))

    print(f"Training set: {len(train_df)} rows saved to {args.train_data}")
    print(f"Testing set: {len(test_df)} rows saved to {args.test_data}")


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
