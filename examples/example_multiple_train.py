import copy
import json
import os
import pathlib

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

from model_eval import model_evaluation

def main():
    #setup dataset
    np.random.seed(3)
    distances = np.linspace(2, 5, 500)
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

    data = model_evaluation.dataset(train_images=images[:400], test_images=images[400:])


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
                "epochs": 400,
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
    #replace "amptorch" below with name of conda env that has AMPTorch installed
    results = model_evaluation.evaluate_models(data, config_dicts=[config_1, config_2], enable_parallel=True, 
                                               workspace=workspace, time_limit="00:30:00", mem_limit=2, 
                                               conda_env="amptorch", num_train_iters=2)

    #print results
    print("Test errors: {}".format([metrics.test_error for metrics in results]))

if __name__ == "__main__":
    main()
