# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("register")
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print(f"Registering model: {args.model_name}")

    # Step 1: Load the model
    print(f"Loading model from: {args.model_path}")
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log the model in MLflow
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=args.model_name)
    print(f"Model {args.model_name} logged successfully.")

    # Step 3: Register the logged model
    # Retrieve the latest version of the registered model
    client = mlflow.tracking.MlflowClient()
    registered_model = client.get_latest_versions(args.model_name, stages=["None"])[0]
    model_uri = registered_model.source
    model_version = registered_model.version

    print(f"Model registered as '{args.model_name}' version {model_version}.")

    # Step 4: Write registration details to JSON
    model_info = {
        "model_name": args.model_name,
        "model_version": model_version,
        "model_uri": model_uri
    }

    os.makedirs(os.path.dirname(args.model_info_output_path), exist_ok=True)
    with open(args.model_info_output_path, "w") as f:
        json.dump(model_info, f, indent=4)

    print(f"Model information written to {args.model_info_output_path}")


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

