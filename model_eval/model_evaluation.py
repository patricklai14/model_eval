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
from ase.io.trajectory import Trajectory

from model_eval import constants, utils

#structure for holding dataset and parameters
class dataset:
    def __init__(self, train_images=None, train_data_files=None, test_images=None, test_data_files=None):
        self.train_images = train_images
        self.train_data_files = train_data_files
        self.test_images = test_images
        self.test_data_files = test_data_files

        if not self.train_images and not self.train_data_files:
            raise RuntimeError("dataset object created with no training data")            

        if self.train_images is not None and self.train_data_files is not None:
            raise RuntimeError("only one of train_images and train_data_files can be set")

        if self.test_images is not None and self.test_data_files is not None:
            raise RuntimeError("only one of test_images and test_data_files can be set")

class evaluation_params:
    def __init__(self, config):
        self.set_default_values()
        self.set_values(config)
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
            constants.CONFIG_EVAL_LOSS_TYPE: "mae",
            constants.CONFIG_RAND_SEED: 1
        }

    def set_values(self, config):
        if constants.CONFIG_AMPTORCH_CONFIG in config:
            for key, value in config[constants.CONFIG_AMPTORCH_CONFIG].items():
                self.amptorch_config[key] = value

        for key, value in config.items():
            if key == constants.CONFIG_AMPTORCH_CONFIG:
                continue

            self.eval_config[key] = value

class model_metrics:
    def __init__(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error

def train_model(eval_params, data, run_dir, checkpoint_dir=""):
    #AMPTorch modifies the config we pass into it. We work around this by making a copy
    eval_params = copy.deepcopy(eval_params)

    #handle datasets
    #TODO: clean this up?
    train_images = None
    if data.train_images:
        eval_params.amptorch_config["dataset"]["raw_data"] = data.train_images
        train_images = data.train_images
    else:
        #check just the first file
        if ".lmdb" in data.train_data_files[0]:
            #handle lmdb files
            data_file_type = "lmdb"
            eval_params.amptorch_config["dataset"]["lmdb_path"] = data.train_data_files

        elif ".p" in data.train_data_files[0]:
            #handle pickled images
            data_file_type = "pickle"
            if len(data.train_data_files) != 1:
                raise RuntimeError("only one pickled training data file allowed")

            train_images = pickle.load(open(data.train_data_files[0], "rb"))
            eval_params.amptorch_config["dataset"]["raw_data"] = train_images

        elif ".traj" in data.train_data_files[0]:
            #handle traj files
            data_file_type = "traj"
            if len(data.train_data_files) != 1:
                raise RuntimeError("only one traj training data file allowed")

            traj = Trajectory(data.train_data_files[0])
            train_images = [img for img in traj]
            eval_params.amptorch_config["dataset"]["raw_data"] = train_images

        else:
            raise RuntimeError("Data files must be .lmdb, .p, or .traj files")

    eval_params.amptorch_config["cmd"]["run_dir"] = run_dir

    trainer = AtomsTrainer(eval_params.amptorch_config)
    if checkpoint_dir:
        trainer.load_pretrained(checkpoint_dir)

    trainer.train()

    #calculate training error
    loss_type = eval_params.eval_config[constants.CONFIG_EVAL_LOSS_TYPE]
    if train_images:
        predictions = trainer.predict(train_images)
        true_energies_train = np.array([image.get_potential_energy() for image in train_images])
        pred_energies = np.array(predictions["energy"])

        #calculate loss
        if loss_type == constants.CONFIG_LOSS_TYPE_MSE:
            curr_error_train = np.mean((true_energies_train - pred_energies) ** 2)
        elif loss_type == constants.CONFIG_LOSS_TYPE_MAE:
            curr_error_train = np.mean(np.abs(true_energies_train - pred_energies))
        elif loss_type == constants.CONFIG_LOSS_TYPE_ATOM_MAE:
            total_errors = np.abs(true_energies_train - pred_energies)
            num_atoms = np.array([img.get_global_number_of_atoms() for img in train_images])
            curr_error_train = np.mean(total_errors / num_atoms)

        print("Train Error: {}".format(curr_error_train))
    else:
        curr_error_train = -1.

    return trainer, curr_error_train

#evaluate model with a single train/test split
def evaluate_model_one_split(eval_params, data, run_dir, checkpoint_dir=""):

    trainer, curr_error_train = train_model(eval_params, data, run_dir, checkpoint_dir)

    if data.test_images:
        test_images = data.test_images
    elif data.test_data_files:
        
        if len(data.test_data_files) != 1:
            raise RuntimeError("only one testing data file allowed")

        if ".p" in data.test_data_files[0]:
            #handle pickled images
            test_images = pickle.load(open(data.test_data_files[0], "rb"))

        elif ".traj" in data.test_data_files[0]:
            #handle traj files
            traj = Trajectory(data.test_data_files[0])
            test_images = [img for img in traj]

        else:
            raise RuntimeError("Test data files must be .p, or .traj files")

    #test MSE
    predictions = trainer.predict(test_images)
    true_energies_test = np.array([image.get_potential_energy() for image in test_images])
    pred_energies = np.array(predictions["energy"])

    #calculate loss
    loss_type = eval_params.eval_config[constants.CONFIG_EVAL_LOSS_TYPE]
    if loss_type == constants.CONFIG_LOSS_TYPE_MSE:
        curr_error_test = np.mean((true_energies_test - pred_energies) ** 2)
    elif loss_type == constants.CONFIG_LOSS_TYPE_MAE:
        curr_error_test = np.mean(np.abs(true_energies_test - pred_energies))
    elif loss_type == constants.CONFIG_LOSS_TYPE_ATOM_MAE:
        total_errors = np.abs(true_energies_test - pred_energies)
        num_atoms = np.array([img.get_global_number_of_atoms() for img in test_images])
        curr_error_test = np.mean(total_errors / num_atoms)

    print("Test Error: {}".format(curr_error_test))
    
    return curr_error_train, curr_error_test

#run model evaluation with given params and return (train_mse, test_mse)
def evaluate_model(eval_config, data, run_dir='./', checkpoint_dir=""):
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

        error_test_list = []
        error_train_list = []

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
                curr_error_train, curr_error_test = evaluate_model_one_split(eval_params, curr_data, run_dir, 
                                                                             checkpoint_dir)
                
                error_test_list.append(curr_error_test)
                error_train_list.append(curr_error_train)

        error_test_avg = np.mean(error_test_list)
        print("Avg test error: {}".format(error_test_avg))

        error_train_avg = np.mean(error_train_list)
        print("Avg train error: {}".format(error_train_avg))

        return error_train_avg, error_test_avg


    else:
        #simple train on training set, test on test set
        error_train, error_test = evaluate_model_one_split(eval_params, data, run_dir, checkpoint_dir)
        return error_train, error_test

#evaluate the models given in config_files and return performance metrics
#save_model_dir = directory to save model checkpoints
def evaluate_models(dataset=None, datasets=[], config_dicts=None, config_files=None, enable_parallel=False, 
                    workspace=None, time_limit="00:30:00", mem_limit=2, conda_env=None, num_train_iters=1,
                    save_model_dir="", checkpoint_dirs=[]):

    #basic error checking on datasets/configs
    if not dataset:
        if not datasets:
            raise RuntimeError("either one of dataset or datasets must be provided")

        #if multiple datasets provided, the number of datasets must match number of configs
        if not config_dicts:
            if not config_files:
                raise RuntimeError("either one of config_dicts or config_files must be provided")

            if len(datasets) != len(config_files):
                raise RuntimeError("length of datasets must match length of config_files")

        else:
            if config_files is not None:
                raise RuntimeError("Both config_dicts and config_files provided - not supported")

            if len(datasets) != len(config_dicts):
                raise RuntimeError("length of datasets must match length of config_dicts")

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

        if dataset is not None:
            #write dataset object to disk
            dataset_file = data_path / constants.DATASET_FILE
            pickle.dump(dataset, open(dataset_file, "wb" ))

        #set up config, dataset files
        if not config_files:
            config_files = []
            job_names = set()
            for config_dict in config_dicts:
                job_name = config_dict[constants.CONFIG_JOB_NAME]
                if job_name in job_names:
                    raise RuntimeError("duplicate job name: {}".format(job_name))

                config_file = config_path / "config_{}.json".format(job_name)
                json.dump(config_dict, open(config_file, "w+"), indent=2)
                config_files.append(config_file)

        model_eval_script_dir = pathlib.Path(__file__).parent.absolute()
        model_eval_script = model_eval_script_dir / constants.EVAL_MODEL_SCRIPT

        job_names = []
        dataset_files = []
        configs = []
        for i in range(len(config_files)):
            config_file = config_files[i]
            config = json.load(open(config_file, "r"))
            job_name = config[constants.CONFIG_JOB_NAME]

            if job_name in job_names:
                raise RuntimeError("duplicate job name: {}".format(job_name))

            #write job-specific dataset to disk if required
            if len(datasets) != 0:
                #write dataset object to disk
                dataset_file = data_path / "{}_{}.p".format(constants.DATASET_FILE_PREFIX, job_name)
                pickle.dump(datasets[i], open(dataset_file, "wb" ))
            dataset_files.append(dataset_file)

            job_names.append(job_name)
            configs.append(config)

        #train for num_train_iters times before evaluating
        for train_iter in range(num_train_iters):
            additional_cmd_options = ""
            if train_iter < num_train_iters - 1:
                additional_cmd_options += " --train_only"

            #create pace pbs files, then submit jobs
            for i in range(len(job_names)):
                job_name = job_names[i]
                dataset_file = dataset_files[i]
                config_file = config_files[i]
                config = configs[i]

                if train_iter == 0 and checkpoint_dirs:
                    additional_cmd_options += " --checkpoint {}".format(checkpoint_dirs[i])

                elif train_iter > 0:
                    additional_cmd_options += " --checkpoint {} --delete_prev_checkpoint".format(
                                                utils.get_checkpoint_dir(training_path / job_name)) 

                command_str = "python {} --workspace {} --job_name {} --data {} --config {}{}".format(
                            model_eval_script, workspace, job_name, dataset_file, config_file, additional_cmd_options)
                pbs_file = utils.create_pbs(pbs_path, job_name, command_str, conda_env, time=time_limit, mem=mem_limit)

                print("Submitting job {} with config: {}".format(job_name, config))
                subprocess.run(["qsub", pbs_file])

            #check for/collect results
            results = []
            for name in job_names:
                if train_iter < num_train_iters - 1:
                    curr_output_file = temp_output_path / "output_{}.txt".format(name)                    
                else:
                    curr_output_file = temp_output_path / "output_{}.json".format(name)

                while not curr_output_file.exists():
                    print("results for job {} not ready. Sleeping for 60s".format(name))
                    print("looking for: {}".format(curr_output_file))
                    time.sleep(60)

                #collect results if this is the last iteration
                if train_iter == num_train_iters - 1:
                    result_dict = json.load(open(curr_output_file, "r"))
                    train_mse = result_dict[constants.TRAIN_MSE]
                    test_mse = result_dict[constants.TEST_MSE]

                    results.append(model_metrics(train_mse, test_mse))
                else:
                    curr_output_file.unlink()

        #if we're saving the model, copy model from workspace
        #TODO: consider saving to this directory directly
        if save_model_dir:
            save_model_dir = pathlib.Path(save_model_dir)
            if save_model_dir.exists() and save_model_dir.is_dir():
                shutil.rmtree(save_model_dir)
            shutil.move(str(training_path), str(save_model_dir))

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
            print("Test Error: {}".format(test_mse))

            results.append(model_metrics(train_mse, test_mse))

        return results

