import numpy as np

import argparse
import json
import os
import pathlib
import pdb
import pickle

import model_evaluation
import constants

def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--workspace", type=str, required=True, 
                        help="top-level workspace")
    parser.add_argument("--job_name", type=str, required=True,
                        help="name of job")
    parser.add_argument("--data", type=str, required=True,
                        help="location of dataset file")
    parser.add_argument("--config", type=str, required=True,
                        help="location of config file")

    args = parser.parse_args()
    args_dict = vars(args)

    workspace_path = pathlib.Path(args_dict["workspace"])
    job_name = args_dict["job_name"]

    #load dataset
    data_file = args_dict["data"]
    dataset = pickle.load(open(data_file, "rb"))

    #load model config
    config_file = args_dict["config"]
    config = json.load(open(config_file, "r"))

    run_dir = workspace_path / "{}/{}".format(constants.TRAINING_DIR, job_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    #get model performance
    print("Evaluating with config: {}".format(config))
    train_mse, test_mse = model_evaluation.evaluate_model(config, dataset, run_dir)
    print("Test MSE: {}".format(test_mse))

    #write result to file
    output_path = workspace_path / constants.OUTPUT_DIR / "output_{}.json".format(job_name)
    json.dump({constants.TRAIN_MSE: train_mse, constants.TEST_MSE: test_mse}, open(output_path, "w+"), indent=2)

if __name__ == "__main__":
    main()

