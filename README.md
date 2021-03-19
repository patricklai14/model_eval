`model_eval` is a Python package that performs batch processing of model evaluation tasks for Atomistic ML models implemented in AMPTorch. The interface allows users to provide a list of highly customizable model configurations and receive performance metrics associated with each provided model. The package supports both sequential and parallel (on PACE) model evaluation and does not require any user intervention for either mode.

### Installation:
1. Activate a conda environment with a working version of AMPTorch installed
2. Install `model_eval` with pip (pointed at the top-level directory):
`pip install -e /path/to/repo/model_eval`

### Running the example scripts
Examples for using `model_eval` are provided in the `examples` directory. For example, to run the example script for MCSH models:
`python examples/example_mcsh.py`

Note: Before running the example script, you will have to make a change to the call to `model_evaluation.evaluate_models()`. The `conda_env` argument is set to `"amptorch"`; this should be changed to the name of the conda environment you are using to run AMPTorch.

To run the example for training on `lmdb` files:
1. generate the sample `lmdb` file by running: `python examples/construct_lmdb.py` (copied from the AMPTorch example code)
2. run the evaluation script: `python examples/example_lmdb.py`

