import copy
import json
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

    data = model_evaluation.dataset(train_images=images[:50], test_images=images[50:])

    #set up evaluation configs
    elements = ["Cu","C","O"]
    dir_prefix = "/storage/home/hpaceice1/plai30/sandbox"
    curr_dir = pathlib.Path(__file__).parent.absolute()

    #note that values in config must be basic python types (e.g. lists instead of np arrays)
    sigmas = [0.049999999999999996, 0.1057371263440564, 0.22360679774997896, 0.4728708045015879, 1.0]

    gmp_params = {
        "MCSHs": {
            "0": {"groups": [1], "sigmas": sigmas},
            "1": {"groups": [1], "sigmas": sigmas},
        },
        "atom_gaussians": {
            "C": str(curr_dir / "MCSH_potential/C_coredensity_5.g"),
            "O": str(curr_dir / "MCSH_potential/O_totaldensity_7.g"),
            "Cu": str(curr_dir / "MCSH_potential/Cu_totaldensity_5.g")
        },
        "cutoff": 8
    }

    config_1 = {
        "name": "test_job_1",
        "evaluation_type": "train_test_split",
        "seed": 1,
        "amptorch_config": {
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
                "epochs": 1000,
            },
            "dataset": {
                #notice that we don't set the actual dataset here
                "val_split": 0,
                "elements": elements,
                "fp_scheme": "mcsh",
                "fp_params": gmp_params,
                "save_fps": False,
                "sampling": {
                    "sampling_method": "nns",
                    "sampling_params": {
                        "cutoff": 0.5,
                        "rate": 0.5,
                        "method": "pykdtree",
                        "start_trial_component": 1,
                        "max_component": 1,
                        "target_variance": 0.999999, 
                    },
                    "save": False,
                },
                "scaling": {
                    "type": "normalize", 
                    "range": (-1, 1),
                    "elementwise": False
                }
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
    config_2["amptorch_config"]["dataset"]["sampling"]["sampling_params"]["rate"] = 0.3

    #run model evaluation
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
