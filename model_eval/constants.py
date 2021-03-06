
#workspace directory names
CONFIG_DIR = "config"
PBS_DIR = "pbs"
OUTPUT_DIR = "output"
TRAINING_DIR = "training"
DATA_DIR = "data"

#workspace file names
DATASET_FILE = "data.p"
DATASET_FILE_PREFIX = "data"

EVAL_MODEL_SCRIPT = "evaluate_single_model_script.py"

#result fields
TRAIN_MSE = "avg_train_mse"
TEST_MSE = "avg_test_mse"

#eval config fields
CONFIG_JOB_NAME = "name"
CONFIG_EVAL_TYPE = "evaluation_type"
CONFIG_EVAL_NUM_FOLDS = "num_folds"
CONFIG_EVAL_CV_ITERS = "cv_iters"
CONFIG_EVAL_LOSS_TYPE = "loss_type"
CONFIG_RAND_SEED = "seed"
CONFIG_FP_SCHEME = "fp_scheme"
CONFIG_AMPTORCH_CONFIG = "amptorch_config"

#evaluation_param_fields
PARAM_MCSH_GROUP_PARAMS = "mcsh_group_params"

#evaluation param values
CONFIG_LOSS_TYPE_MAE = "mae"
CONFIG_LOSS_TYPE_MSE = "mse"
CONFIG_LOSS_TYPE_ATOM_MAE = "atom_mae"
