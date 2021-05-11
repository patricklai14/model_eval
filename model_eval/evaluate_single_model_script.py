import numpy as np

import argparse
import json
import os
import pathlib
import pdb
import pickle
import shutil

from model_eval import model_evaluation, constants, utils

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
    parser.add_argument("--checkpoint", type=str, required=False, default="str",
                        help="if provided, load model from this checkpoint")
    parser.add_argument("--train_only", action="store_true", 
                        help="train without evaluating the model")
    parser.add_argument("--delete_prev_checkpoint", action="store_true",
                        help="delete provided checkpoint after training")

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
    run_dir.mkdir(parents=True, exist_ok=True)
    
    #check if we're loading a model from checkpoint
    checkpoint_dir = args_dict["checkpoint"]

    #train and/or get model performance 
    print("Evaluating with config: {}".format(config))
    if args_dict["train_only"]:
        eval_params = model_evaluation.evaluation_params(config)
        model_evaluation.train_model(eval_params, dataset, run_dir, checkpoint_dir)
        output_path = workspace_path / constants.OUTPUT_DIR / "output_{}.txt".format(job_name)
        output_file = open(output_path, "w")
        output_file.write("job {} complete".format(job_name))
        output_file.close()

    else:
        train_mse, test_mse = model_evaluation.evaluate_model(config, dataset, run_dir, checkpoint_dir)
        print("Test Error: {}".format(test_mse))

        #write result to file
        output_path = workspace_path / constants.OUTPUT_DIR / "output_{}.json".format(job_name)
        json.dump({constants.TRAIN_MSE: train_mse, constants.TEST_MSE: test_mse}, open(output_path, "w+"), indent=2)

    #delete last checkpoint directory if necessary
    if checkpoint_dir and args_dict["delete_prev_checkpoint"]:
        shutil.rmtree(checkpoint_dir) 

if __name__ == "__main__":
    main()

