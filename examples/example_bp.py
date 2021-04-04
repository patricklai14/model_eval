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

    elements = ["Cu","C","O"]
    data = model_evaluation.dataset(images)

    #create configs
    Gs_1 = {"default": {
                "G2": {
                    "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4).tolist(),
                    "rs_s": [0]},
                "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
                "cutoff": 6,
            }}

    Gs_2 = copy.deepcopy(Gs_1)
    Gs_2["default"]["cutoff"] = 2

    config_1 = {
        "name": "test_job_1",
        "evaluation_type": "k_fold_cv",
        "cv_iters": 2,
        "num_folds": 2,
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
                "fp_scheme": "gaussian",
                "fp_params": Gs_1,
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

    config_2 = copy.deepcopy(config_1)
    config_2["name"] = "test_job_2"
    config_2["amptorch_config"]["dataset"]["fp_params"] = Gs_2

    #run model evaluation
    curr_dir = pathlib.Path(__file__).parent.absolute()
    workspace = curr_dir / "test_workspace"

    #running model evaluation by passing in a list of config dicts
    #replace "amptorch" below with name of conda env that has AMPTorch installed
    results = model_evaluation.evaluate_models(data, config_dicts=[config_1, config_2], 
                                               enable_parallel=True, workspace=workspace, 
                                               time_limit="00:30:00", mem_limit=2, conda_env="amptorch")

    #print results
    print("CV errors: {}".format([metrics.test_error for metrics in results]))

if __name__ == "__main__":
    main()
