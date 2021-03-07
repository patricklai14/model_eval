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

import constants
import utils

#structure for holding dataset and parameters
class dataset:
    def __init__(self, elements, train_images, test_images=None, atom_gaussians=None):
        self.elements = elements
        self.train_images = train_images
        self.test_images = test_images
        self.atom_gaussians = atom_gaussians

#model/evaluation parameters
class evaluation_params:
    def __init__(self, config):
        utils.validate_model_eval_params(config)
        self.set_default_params()
        self.set_config_params(config)

    def set_default_params(self):
        self.params = {constants.CONFIG_NN_LAYERS: constants.DEFAULT_NN_LAYERS,
                       constants.CONFIG_NN_NODES: constants.DEFAULT_NN_NODES,
                       constants.CONFIG_NN_LR: constants.DEFAULT_NN_LR,
                       constants.CONFIG_NN_BATCH_SIZE: constants.DEFAULT_NN_BATCH_SIZE,
                       constants.CONFIG_NN_EPOCHS: constants.DEFAULT_NN_EPOCHS,
                       constants.CONFIG_EVAL_NUM_FOLDS: constants.DEFAULT_EVAL_NUM_FOLDS,
                       constants.CONFIG_EVAL_CV_ITERS: constants.DEFAULT_EVAL_CV_ITERS,
                       constants.CONFIG_RAND_SEED: constants.DEFAULT_RAND_SEED}

    def set_config_params(self, config):
        for key, value in config.items():
            if key == constants.CONFIG_GROUPS_BY_ORDER:
                mcsh_group_params = copy.deepcopy(value)
                for order, group_params in mcsh_group_params.items():
                    group_params[constants.CONFIG_SIGMAS] = np.array(config[constants.CONFIG_SIGMAS])

                self.params[constants.PARAM_MCSH_GROUP_PARAMS] = mcsh_group_params
                continue

            self.params[key] = value

class model_metrics:
    def __init__(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error

#evaluate model with a single train/test split
def evaluate_model_one_split(eval_params, data, run_dir):
    if eval_params.params[constants.CONFIG_FP_TYPE] == "mcsh":
        fp_scheme = "mcsh"
        fp_params = {"MCSHs": eval_params.params[constants.PARAM_MCSH_GROUP_PARAMS],
                     "atom_gaussians": data.atom_gaussians,
                     "cutoff": eval_params.params[constants.CONFIG_CUTOFF]
                    }

    else:
        fp_scheme = "bp"
        fp_params = eval_params.params[constants.CONFIG_BP_PARAMS]

    config = {
        "model": {"get_forces": False, 
                  "num_layers": eval_params.params[constants.CONFIG_NN_LAYERS], 
                  "num_nodes": eval_params.params[constants.CONFIG_NN_NODES]},
        "optim": {
            "device": "cpu",
            "force_coefficient": 0.0,
            "lr": eval_params.params[constants.CONFIG_NN_LR],
            "batch_size": eval_params.params[constants.CONFIG_NN_BATCH_SIZE],
            "epochs": eval_params.params[constants.CONFIG_NN_EPOCHS],
        },
        "dataset": {
            "raw_data": data.train_images,
            "val_split": 0,
            "elements": data.elements,
            "fp_scheme": fp_scheme,
            "fp_params": fp_params,
            "save_fps": False,
        },
        "cmd": {
            "debug": False,
            "run_dir": run_dir,
            "seed": eval_params.params[constants.CONFIG_RAND_SEED],
            "identifier": "test",
            "verbose": False,
            "logger": False,
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    #test MSE
    predictions = trainer.predict(data.test_images)
    true_energies_test = np.array([image.get_potential_energy() for image in data.test_images])
    pred_energies = np.array(predictions["energy"])
    curr_mse_test = np.mean((true_energies_test - pred_energies) ** 2)
    print("Test MSE:", curr_mse_test)

    #train MSE
    predictions = trainer.predict(data.train_images)
    true_energies_train = np.array([image.get_potential_energy() for image in data.train_images])
    pred_energies = np.array(predictions["energy"])
    curr_mse_train = np.mean((true_energies_train - pred_energies) ** 2)
    print("Train MSE:", curr_mse_train)

    return curr_mse_train, curr_mse_test

#run model evaluation with given params and return (train_mse, test_mse)
def evaluate_model(eval_config, data, run_dir='./'):
    eval_params = evaluation_params(eval_config)
    np.random.seed(eval_params.params[constants.CONFIG_RAND_SEED])

    if eval_params.params[constants.CONFIG_EVAL_TYPE] == "k_fold_cv":

        #setup for k-fold cross validation
        num_folds = eval_params.params[constants.CONFIG_EVAL_NUM_FOLDS]

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
        num_cv_iters = eval_params.params[constants.CONFIG_EVAL_CV_ITERS]

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

                curr_data = dataset(data.elements, images_train, images_test, data.atom_gaussians)
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
        mse_train, mse_test = evaluate_model_one_split(eval_params, data)
        return mse_train, mse_test

#evaluate the models given in config_files and return performance metrics
def evaluate_models(dataset, config_dicts=None, config_files=None, eval_mode="cv", enable_parallel=False, workspace=None):

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
            pbs_file = utils.create_pbs(pbs_path, job_name, command_str, time="00:30:00")

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
