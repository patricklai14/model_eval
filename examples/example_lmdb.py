import copy
import json
import pathlib

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

from model_eval import model_evaluation

def main():
    curr_dir = pathlib.Path(__file__).parent.absolute()

    #setup dataset
    #train data
    train_data_file = str(curr_dir / "data/data_train.lmdb")

    #test data
    np.random.seed(3)
    distances = np.linspace(2, 5, 100)
    images = []
    for i in range(len(distances)):
        l = distances[i]
        image = Atoms(
            "CuCO",
            [
                (-l * np.sin(0.65), l * np.cos(0.65), 0),
                (0, 0, 0),
                (l * np.sin(0.65), l * np.cos(0.65), 0)
            ],
        )

        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        images.append(image)

    elements = ["Cu","C","O"]
    data = model_evaluation.dataset(train_data_files=[train_data_file], test_images=images)

    #create configs
    config_1 = {
        "name": "test_job_1",
        "evaluation_type": "train_test_split",
        "seed": 1,
        "amptorch_config": {
            "model": {
                "get_forces": True, 
                "num_layers": 2, 
                "num_nodes": 3},
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
                "elements": elements
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
    config_2["amptorch_config"]["model"]["num_layers"] = 3
    config_2["amptorch_config"]["model"]["num_nodes"] = 20

    #run model evaluation
    workspace = curr_dir / "test_workspace"
    results = model_evaluation.evaluate_models(data, config_dicts=[config_1, config_2], 
                                               enable_parallel=True, workspace=workspace, 
                                               time_limit="00:30:00", mem_limit=2, conda_env="amptorch")

    #print results
    print("test errors: {}".format([metrics.test_error for metrics in results]))

if __name__ == "__main__":
    main()
