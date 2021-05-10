import copy
import json
import os
import pathlib

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from amptorch.trainer import AtomsTrainer

from model_eval import model_evaluation, utils

def main():
    #setup dataset
    np.random.seed(3)
    distances = np.linspace(2, 5, 100)
    images = []
    for i in range(len(distances)):
        l = distances[i]
        image = Atoms(
            "CuCO",
            [
                (-l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0)),
                (0, 0, 0),
                (l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0))
            ],
        )

        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        images.append(image)

    train_images = images[:80]
    test_images = images[80:]
    data = model_evaluation.dataset(train_images=train_images, test_images=test_images)


    elements = ["Cu","C","O"]
    dir_prefix = "/storage/home/hcoda1/7/plai30/sandbox"
    atom_gaussians = {"C": os.path.join(dir_prefix, "config/MCSH_potential/C_coredensity_5.g"),
                      "O": os.path.join(dir_prefix, "config/MCSH_potential/O_totaldensity_7.g"),
                      "Cu": os.path.join(dir_prefix, "config/MCSH_potential/Cu_totaldensity_5.g")}

    #note that values in config must be basic python types (e.g. lists instead of np arrays)
    sigmas = [0.049999999999999996, 0.1057371263440564, 0.22360679774997896, 0.4728708045015879, 1.0]

    gmp_params_1 = {
        "MCSHs": {
            "0": {"groups": [1], "sigmas": sigmas},
            "1": {"groups": [1], "sigmas": sigmas},
        },
        "atom_gaussians": {
            "C": os.path.join(dir_prefix, "config/MCSH_potential/C_coredensity_5.g"),
            "O": os.path.join(dir_prefix, "config/MCSH_potential/O_totaldensity_7.g"),
            "Cu": os.path.join(dir_prefix, "config/MCSH_potential/Cu_totaldensity_5.g"),
        },
        "cutoff": 8
    }

    config_1 = {
        "name": "test_job_1",
        "evaluation_type": "train_test_split",
        "loss_type": "mse",
        "seed": 1,
        "amptorch_config": {
            "model": {
                "get_forces": False, 
                "num_layers": 3, 
                "num_nodes": 20},
            "optim": {
                "device": "cpu",
                "force_coefficient": 0.0,
                "lr": 1e-3,
                "batch_size": 32,
                "epochs": 1000,
            },
            "dataset": {
                #notice that we don't set the actual dataset here
                "val_split": 0,
                "elements": elements,
                "fp_scheme": "mcsh",
                "fp_params": gmp_params_1,
                "save_fps": False,
            },
            "cmd": {
                "debug": False,
                "run_dir": "./",
                "seed": 1,
                "identifier": "test",
                "verbose": False,
                "logger": False,
            }
        }
    }

    gmp_params_2 = copy.deepcopy(gmp_params_1)
    gmp_params_2["MCSHs"] = {
        "0": {"groups": [1], "sigmas": sigmas},
        "1": {"groups": [1], "sigmas": sigmas},
        "2": {"groups": [1, 2], "sigmas": sigmas}   
    }

    config_2 = copy.deepcopy(config_1)
    config_2["name"] = "test_job_2"
    config_2["amptorch_config"]["dataset"]["fp_params"] = gmp_params_2

    curr_dir = pathlib.Path(__file__).parent.absolute()

    config_file_1 = curr_dir / "config_1.json" 
    json.dump(config_1, open(config_file_1, "w+"), indent=2)

    config_file_2 = curr_dir / "config_2.json"
    json.dump(config_2, open(config_file_2, "w+"), indent=2)


    #run model evaluation
    workspace = curr_dir / "test_workspace"
    save_model_dir = curr_dir / "saved_models"
    
    #running model evaluation by passing in a list of config files
    #results = model_evaluation.evaluate_models(data, config_dicts=[str(config_file_1), str(config_file_2)],
    #                                           enable_parallel=True, workspace=workspace,
    #                                           time_limit="00:30:00", mem_limit=2, conda_env="amptorch")

    #running model evaluation by passing in a list of config dicts
    #replace "amptorch" below with name of conda env that has AMPTorch installed
    config_dicts = [config_1, config_2]
    results = model_evaluation.evaluate_models(data, config_dicts=config_dicts, 
                                               enable_parallel=True, workspace=workspace, 
                                               time_limit="00:30:00", mem_limit=2, conda_env="amptorch",
                                               save_model_dir=save_model_dir)

    #print results
    test_errors = [metrics.test_error for metrics in results]
    print("Test errors: {}".format(test_errors))

    #test if saved models produce matching results
    saved_test_errors = []
    true_energies_test = np.array([image.get_potential_energy() for image in test_images])
    for i in range(2):
        amptorch_config = config_dicts[i]["amptorch_config"]
        amptorch_config["dataset"]["raw_data"] = test_images #not used, but required by amptorch
        trainer = AtomsTrainer(amptorch_config)

        checkpoint_dir = utils.get_checkpoint_dir(save_model_dir / "test_job_{}".format(i + 1))
        trainer.load_pretrained(checkpoint_dir)
        predictions = trainer.predict(test_images)
        pred_energies = np.array(predictions["energy"])
    
        curr_error_test = np.mean((true_energies_test - pred_energies) ** 2)
        saved_test_errors.append(curr_error_test)
    print("Test errors for saved models: {}".format(saved_test_errors))


if __name__ == "__main__":
    main()
