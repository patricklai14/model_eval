import copy
import json
import pathlib

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

from model_eval import model_evaluation

def main():
    #create configs
    config_1 = {"name": "test_job_1",
                "evaluation_type": "train_test_split",
                "nn_layers": 2,
                "nn_nodes": 3,
                "nn_learning_rate": 1e-3,
                "nn_batch_size": 32,
                "nn_epochs": 1000,
                "seed": 1}

    config_2 = {"name": "test_job_2",
                "evaluation_type": "train_test_split",
                "nn_layers": 3,
                "nn_nodes": 20,
                "nn_learning_rate": 1e-3,
                "nn_batch_size": 32,
                "nn_epochs": 1000,
                "seed": 1}

    curr_dir = pathlib.Path(__file__).parent.absolute()
    dir_prefix = "/storage/home/hpaceice1/plai30/sandbox"

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
    atom_gaussians = {"C": str(curr_dir / "MCSH_potential/C_coredensity_5.g"),
                      "O": str(curr_dir / "MCSH_potential/O_totaldensity_7.g"),
                      "Cu": str(curr_dir / "MCSH_potential/Cu_totaldensity_5.g")}
    data = model_evaluation.dataset(elements, train_data_files=[train_data_file], test_images=images, 
                                    atom_gaussians=atom_gaussians)

    #run model evaluation
    workspace = curr_dir / "test_workspace"
    results = model_evaluation.evaluate_models(data, config_dicts=[config_1, config_2], 
                                               enable_parallel=True, workspace=workspace, 
                                               time_limit="00:30:00", mem_limit=2, conda_env="amptorch")

    #print results
    print("test errors: {}".format([metrics.test_error for metrics in results]))

if __name__ == "__main__":
    main()
