import json
import os
import pathlib

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT

import model_evaluation

def main():
    #create config files
    sigmas = [0.049999999999999996, 0.1057371263440564, 0.22360679774997896, 0.4728708045015879, 1.0]
    config_1 = {"name": "test_job_1",
                "fingerprint_type": "mcsh",
                "evaluation_type": "k_fold_cv",
                "num_folds": 2,
                "cv_iters": 2,
                "cutoff": 8,
                "groups_by_order": {
                    "0": {"groups": [1]},
                    "1": {"groups": [1]}},
                "sigmas": sigmas}

    config_2 = {"name": "test_job_2",
                "fingerprint_type": "mcsh",
                "evaluation_type": "k_fold_cv",
                "num_folds": 2,
                "cv_iters": 2,
                "cutoff": 8,
                "groups_by_order": {
                    "0": {"groups": [1]},
                    "1": {"groups": [1]},
                    "2": {"groups": [1, 2]}},
                "sigmas": sigmas}

    curr_dir = pathlib.Path(__file__).parent.absolute()

    config_file_1 = curr_dir / "config_1.json" 
    json.dump(config_1, open(config_file_1, "w+"), indent=2)

    config_file_2 = curr_dir / "config_2.json"
    json.dump(config_2, open(config_file_2, "w+"), indent=2)

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
    dir_prefix = "/storage/home/hpaceice1/plai30/sandbox"
    atom_gaussians = {"C": os.path.join(dir_prefix, "config/MCSH_potential/C_coredensity_5.g"),
                      "O": os.path.join(dir_prefix, "config/MCSH_potential/O_totaldensity_7.g"),
                      "Cu": os.path.join(dir_prefix, "config/MCSH_potential/Cu_totaldensity_5.g")}
    data = model_evaluation.dataset(elements, images, atom_gaussians=atom_gaussians)

    #run model evaluation
    workspace = curr_dir / "test_workspace"
    results = model_evaluation.evaluate_models(data, config_files=[str(config_file_1), str(config_file_2)], 
                                         enable_parallel=True, workspace=workspace)

    #print results
    print("CV errors: {}".format([metrics.test_error for metrics in results]))

if __name__ == "__main__":
    main()
