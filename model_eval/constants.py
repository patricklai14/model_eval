
#workspace directory names
CONFIG_DIR = "config"
PBS_DIR = "pbs"
OUTPUT_DIR = "output"
TRAINING_DIR = "training"
DATA_DIR = "data"

#workspace file names
TRAIN_DATA_FILE = "train_data.p"
TEST_DATA_FILE = "test_data.p"

EVAL_MODEL_SCRIPT = "evaluate_single_model_script.py"

#result fields
TRAIN_MSE = "avg_train_mse"
TEST_MSE = "avg_test_mse"

#config fields
CONFIG_JOB_NAME = "name"
CONFIG_FP_TYPE = "fingerprint_type"
CONFIG_EVAL_TYPE = "evaluation_type"
CONFIG_CUTOFF = "cutoff"
CONFIG_SIGMAS = "sigmas"
CONFIG_GROUPS_BY_ORDER = "groups_by_order"
CONFIG_BP_PARMS = "bp_params"
CONFIG_NN_LAYERS = "nn_layers"
CONFIG_NN_NODES = "nn_nodes"
CONFIG_NN_LR = "learning_rate"
CONFIG_NN_BATCH_SIZE = "batch_size"
CONFIG_NN_EPOCHS = "epochs"
CONFIG_EVAL_NUM_FOLDS = "num_folds"
CONFIG_EVAL_CV_ITERS = "cv_iters"
CONFIG_RAND_SEED = "seed"

#evaluation_param_fields
PARAM_MCSH_GROUP_PARAMS = "mcsh_group_params"

#default parameters
DEFAULT_NN_LAYERS = 3
DEFAULT_NN_NODES = 20
DEFAULT_NN_LR = 1e-3
DEFAULT_NN_BATCH_SIZE = 32
DEFAULT_NN_EPOCHS = 1000
DEFAULT_EVAL_CV_ITERS = 3
DEFAULT_EVAL_NUM_FOLDS = 5
DEFAULT_RAND_SEED = 1