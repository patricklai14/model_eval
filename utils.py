import pathlib

import constants

#mem is in terms of GB
def create_pbs(location, job_name, command, mem=2, time="00:30:00"):
    pbs_filename = pathlib.Path(location) / "{}.pbs".format(job_name)
    pbs_file = open(pbs_filename, "w+")
    
    pbs_file.write("#PBS -N {}\n".format(job_name))
    pbs_file.write("#PBS -l nodes=1:ppn=1\n")
    pbs_file.write("#PBS -l pmem={}gb\n".format(mem))
    pbs_file.write("#PBS -l walltime={}\n".format(time))
    pbs_file.write("#PBS -q pace-ice\n")
    pbs_file.write("#PBS -j oe\n")
    pbs_file.write("#PBS -o {}\n".format(pathlib.Path(location) / "{}.out".format(job_name)))
    pbs_file.write("\n")

    pbs_file.write("module load anaconda3/2019.10\n")
    pbs_file.write("conda activate amptorch\n")
    pbs_file.write(command)
    pbs_file.write("\n")

    pbs_file.close()

    return pbs_filename

def validate_model_eval_params(params):
    required_fields = [constants.CONFIG_CUTOFF, 
                       constants.CONFIG_FP_TYPE, 
                       constants.CONFIG_EVAL_TYPE]

    for field in required_fields:
        if field not in params:
            raise RuntimeError("required field {} not in config".format(field))

    if params[constants.CONFIG_FP_TYPE] == "mcsh":
        if constants.CONFIG_GROUPS_BY_ORDER not in params or constants.CONFIG_SIGMAS not in params:
            raise RuntimeError("incomplete information in config for MCSH")

            #TODO: validate mcsh parameters

    elif config[constants.CONFIG_FP_TYPE] == "bp":
        if constants.BP_PARAMS not in params:
            raise RuntimeError("bp_params required in config for BP")

        #TODO: validate bp parameters
    
    else:
        raise RuntimeError("invalid fingerprint type: {}".format(params[constants.CONFIG_FP_TYPE]))

#return dict with model eval params with given values
#TODO: perform validation?
def get_model_eval_params(name, fp_type, eval_type, cutoff=None, sigmas=None, groups_by_order=None, bp_params=None, 
                          nn_layers=None, nn_nodes=None, nn_learning_rate=None, nn_batch_size=None, nn_epochs=None,
                          eval_num_folds=None, eval_cv_iters=None, rand_seed=None):
    
    #map keys in dict to arguments
    config_dict = {constants.CONFIG_JOB_NAME: name,
                   constants.CONFIG_FP_TYPE: fp_type,
                   constants.CONFIG_EVAL_TYPE: eval_type,
                   constants.CONFIG_CUTOFF: cutoff,
                   constants.CONFIG_SIGMAS: sigmas,
                   constants.CONFIG_GROUPS_BY_ORDER: groups_by_order,
                   constants.CONFIG_BP_PARMS: bp_params,
                   constants.CONFIG_NN_LAYERS: nn_layers,
                   constants.CONFIG_NN_NODES: nn_nodes,
                   constants.CONFIG_NN_LR: nn_learning_rate,
                   constants.CONFIG_NN_BATCH_SIZE: nn_batch_size,
                   constants.CONFIG_NN_EPOCHS: nn_epochs,
                   constants.CONFIG_EVAL_NUM_FOLDS: eval_num_folds,
                   constants.CONFIG_EVAL_CV_ITERS: eval_cv_iters,
                   constants.CONFIG_RAND_SEED: rand_seed}

    #remove unset entries
    to_remove = []
    for key, value in config_dict.items():
        if value is None:
            to_remove.append(key)

    for key in to_remove:
        del config_dict[key]

    validate_model_eval_params(config_dict)

    return config_dict


