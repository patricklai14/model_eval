import pathlib

from model_eval import constants

#mem is in terms of GB
def create_pbs(location, job_name, command, conda_env, mem=2, time="00:30:00"):
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
    pbs_file.write("conda activate {}\n".format(conda_env))
    pbs_file.write(command)
    pbs_file.write("\n")

    pbs_file.close()

    return pbs_filename

def validate_model_eval_params(params):
    required_fields = [constants.CONFIG_FP_TYPE, 
                       constants.CONFIG_EVAL_TYPE]

    for field in required_fields:
        if field not in params:
            raise RuntimeError("required field {} not in config".format(field))

    if params[constants.CONFIG_FP_TYPE] == "mcsh":
        if constants.CONFIG_MCSH_GROUPS not in params or constants.CONFIG_SIGMAS not in params:
            raise RuntimeError("incomplete information in config for MCSH")

            #TODO: validate mcsh parameters

    elif params[constants.CONFIG_FP_TYPE] == "gaussian":
        if constants.CONFIG_BP_PARAMS not in params:
            raise RuntimeError("bp_params required in config for BP")

        #TODO: validate bp parameters
    
    else:
        raise RuntimeError("invalid fingerprint type: {}".format(params[constants.CONFIG_FP_TYPE]))
