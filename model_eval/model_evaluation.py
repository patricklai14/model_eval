import copy
import json
import pathlib
import pdb
import pickle
import shutil
import subprocess
import time

import numpy as np

from amptorch.trainer import AtomsTrainer

from model_eval import constants, utils

#structure for holding dataset and parameters
class dataset:
    def __init__(self, train_images=None, train_data_files=None, test_images=None):
        self.train_images = train_images
        self.train_data_files = train_data_files
        self.test_images = test_images

        if not self.train_images and not self.train_data_files:
            raise RuntimeError("dataset object created with no training data")            

        if self.train_images is not None and train_data_files is not None:
            raise RuntimeError("only one of train_images and train_data_files can be set")

class evaluation_params:
    def __init__(self, config):
        self.set_default_values()
        self.set_config_params(config)
        utils.validate_amptorch_config(self.amptorch_config)
        utils.validate_eval_config(self.eval_config)

    def set_default_values(self):
        self.amptorch_config = {
            "model": {
                "get_forces": False,
                "num_layers": 3, 
                "num_nodes": 20
            },
            "optim": {
                "device": "cpu",
                "force_coefficient": 0.0,
                "lr": 1e-3,
                "batch_size": 32,
                "epochs": 1000
            },
            "dataset": {
                "val_split": 0,
                "save_fps": False
            },
            "cmd": {
                "debug": False,
                "run_dir": "./",
                "seed": 1,
                "identifier": "test",
                "verbose": False,
                "logger": False
            }
        }

        self.eval_config = {
            constants.CONFIG_EVAL_NUM_FOLDS: 5,
            constants.CONFIG_EVAL_CV_ITERS: 3,
            constants.CONFIG_RAND_SEED: 1
        }

    def set_values(self, config):
        if constants.CONFIG_AMPTORCH_CONFIG in config:
            for key, value in config[constants.CONFIG_AMPTORCH_CONFIG].items():
                self.amptorch_config[key] = value

        for key, value in config:
            if key == constants.CONFIG_AMPTORCH_CONFIG:
                continue

            self.eval_config[key] = value

class model_metrics:
    def __init__(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error

#evaluate model with a single train/test split
def evaluate_model_one_split(eval_params, data, run_dir):

    #TODO: clean this up?
    if data.train_images:
        eval_params.amptorch_config["dataset"]["raw_data"] = data.train_images
    else:
        eval_params.amptorch_config["dataset"]["lmdb_path"] = data.train_data_files

    eval_params.amptorch_config["cmd"]["run_dir"] = run_dir


    trainer = AtomsTrainer(eval_params.amptorch_config)
    trainer.train()

    #test MSE
    predictions = trainer.predict(data.test_images)
    true_energies_test = np.array([image.get_potential_energy() for image in data.test_images])
    pred_energies = np.array(predictions["energy"])
    curr_mse_test = np.mean((true_energies_test - pred_energies) ** 2)
    print("Test MSE:", curr_mse_test)

    #train MSE
    if data.train_images:
        predictions = trainer.predict(data.train_images)
        true_energies_train = np.array([image.get_potential_energy() for image in data.train_images])
        pred_energies = np.array(predictions["energy"])
        curr_mse_train = np.mean((true_energies_train - pred_energies) ** 2)
        print("Train MSE:", curr_mse_train)
    else:
        curr_mse_train = -1.
    
    return curr_mse_train, curr_mse_test

#run model evaluation with given params and return (train_mse, test_mse)
def evaluate_model(eval_config, data, run_dir='./'):
    eval_params = evaluation_params(eval_config)
    np.random.seed(eval_params.eval_config[constants.CONFIG_RAND_SEED])

    if eval_params.eval_config[constants.CONFIG_EVAL_TYPE] == "k_fold_cv":
        #CV only works if training images are provided explicitly
        if not data.train_images:
            raise RuntimeError("training images must be provided explicitly for CV")

        #setup for k-fold cross validation
        num_folds = eval_params.eval_config[constants.CONFIG_EVAL_NUM_FOLDS]

        #Separate data into k folds
        #The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
        #other folds have size n_samples // n_splits, where n_samples is the number of samples.
        fold_indices = []
        num_larger_sets = len(data.train_images) % num_folds
        smaller_set_size = len(data.train_images) // num_folds
        larger_set_size = smaller_set_size + 1
        for i in range(num_folds):
            if i == 0:
                start_index = 0
            else:
                start_index = fold_indices[-1][1]

            if i < num_larger_sets:
                end_index = start_index + larger_set_size
            else:
                end_index = start_index + smaller_set_size

            fold_indices.append((start_index, end_index))

        #number of times to run CV
        num_cv_iters = eval_params.eval_config[constants.CONFIG_EVAL_CV_ITERS]

        mse_test_list = []
        mse_train_list = []

        for _ in range(num_cv_iters):
            np.random.shuffle(data.train_images)
            true_energies = []
            for img in data.train_images:
                true_energies.append(img.get_potential_energy())

            #run k-fold cross validation
            for start_index, end_index in fold_indices:
                images_train = data.train_images[:start_index] + data.train_images[end_index:]
                images_test = data.train_images[start_index:end_index]

                curr_data = dataset(train_images=images_train, test_images=images_test)
                curr_mse_train, curr_mse_test = evaluate_model_one_split(eval_params, curr_data, run_dir)
                
                mse_test_list.append(curr_mse_test)
                mse_train_list.append(curr_mse_train)

        mse_test_avg = np.mean(mse_test_list)
        print("Avg test MSE: {}".format(mse_test_avg))

        mse_train_avg = np.mean(mse_train_list)
        print("Avg train MSE: {}".format(mse_train_avg))

        return mse_train_avg, mse_test_avg


    else:
        #simple train on training set, test on test set
        mse_train, mse_test = evaluate_model_one_split(eval_params, data, run_dir)
        return mse_train, mse_test

#evaluate the models given in config_files and return performance metrics
def evaluate_models(dataset, config_dicts=None, config_files=None, eval_mode="cv", enable_parallel=False, workspace=None,
                    time_limit="00:30:00", mem_limit=2, conda_env=None):

    if enable_parallel:
        if not workspace:
            raise RuntimeError("workspace cannot be None in parallel mode")

        #prepare directories
        workspace_path = pathlib.Path(workspace)
        print("Parallel processing enabled. Initializing workspace in {}".format(workspace_path))
        workspace_subdirs = []
        
        config_path = workspace_path / constants.CONFIG_DIR
        workspace_subdirs.append(config_path)

        pbs_path = workspace_path / constants.PBS_DIR
        workspace_subdirs.append(pbs_path)

        temp_output_path = workspace_path / constants.OUTPUT_DIR
        workspace_subdirs.append(temp_output_path)

        training_path = workspace_path / constants.TRAINING_DIR
        workspace_subdirs.append(training_path)

        data_path = workspace_path / constants.DATA_DIR
        workspace_subdirs.append(data_path)

        for subdir in workspace_subdirs:
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)
            
            subdir.mkdir(parents=True, exist_ok=False)

        #write data to disk
        train_data_file = data_path / constants.TRAIN_DATA_FILE
        pickle.dump(dataset, open(train_data_file, "wb" ))

        #set up config files
        if not config_dicts:
            if not config_files:
                raise RuntimeError("One of config_dicts or config_files must be provided evaluate models")

        else:
            if config_files is not None:
                raise RuntimeError("Both config_dicts and config_files provided - not supported")

            config_files = []
            job_names = set()
            for config_dict in config_dicts:
                job_name = config_dict[constants.CONFIG_JOB_NAME]
                if job_name in job_names:
                    raise RuntimeError("duplicate job name: {}".format(job_name))

                config_file = config_path / "config_{}.json".format(job_name)
                json.dump(config_dict, open(config_file, "w+"), indent=2)
                config_files.append(config_file)

        #create pace pbs files
        job_info = {} #job_name -> (config, pbs_file)
        job_names = []
        model_eval_script_dir = pathlib.Path(__file__).parent.absolute()

        for config_file in config_files:
            config = json.load(open(config_file, "r"))
            job_name = config[constants.CONFIG_JOB_NAME]

            if job_name in job_info:
                raise RuntimeError("duplicate job name: {}".format(job_name))

            model_eval_script = model_eval_script_dir / constants.EVAL_MODEL_SCRIPT
            command_str = "python {} --workspace {} --job_name {} --data {} --config {}".format(
                            model_eval_script, workspace, job_name, train_data_file, config_file)
            pbs_file = utils.create_pbs(pbs_path, job_name, command_str, conda_env, time=time_limit, mem=mem_limit)

            job_info[job_name] = (config, pbs_file)
            job_names.append(job_name)

        #submit jobs on pace
        for name, (config, pbs_file) in job_info.items():
            print("Submitting job {} with config: {}".format(name, config))
            subprocess.run(["qsub", pbs_file])

        #collect results
        results = []
        for name in job_names:
            curr_output_file = temp_output_path / "output_{}.json".format(name)
            while not curr_output_file.exists():
                print("results for job {} not ready. Sleeping for 20s".format(name))
                print("looking for: {}".format(curr_output_file))
                time.sleep(20)

            result_dict = json.load(open(curr_output_file, "r"))
            train_mse = result_dict[constants.TRAIN_MSE]
            test_mse = result_dict[constants.TEST_MSE]

            results.append(model_metrics(train_mse, test_mse))

        #clear workspace
        if workspace_path.exists() and workspace_path.is_dir():
            shutil.rmtree(workspace_path) 

        return results

    else:
        #run sequentially

        #get config dicts if needed
        if not config_dicts:
            if not config_files:
                raise RuntimeError("One of config_dicts or config_files must be provided evaluate models")

            config_dicts = [json.load(open(config_file, "r")) for config_file in config_files]
        else:
            if config_files is not None:
                raise RuntimeError("Both config_dicts and config_files provided - not supported")

        results = []
        for config in config_dicts:
            #get model performance
            print("Evaluating with config: {}".format(config))
            train_mse, test_mse = evaluate_model(config, copy.deepcopy(dataset))
            print("Test MSE: {}".format(test_mse))

            results.append(model_metrics(train_mse, test_mse))

        return results

