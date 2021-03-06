import copy
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
    pbs_file.write("#PBS -q inferno\n")
    pbs_file.write("#PBS -A GT-amedford6\n")
    pbs_file.write("#PBS -j oe\n")
    pbs_file.write("#PBS -o {}\n".format(pathlib.Path(location) / "{}.out".format(job_name)))
    pbs_file.write("\n")

    pbs_file.write("module load anaconda3/2020.02\n")
    pbs_file.write("conda activate {}\n".format(conda_env))
    pbs_file.write(command)
    pbs_file.write("\n")

    pbs_file.close()

    return pbs_filename

def validate_amptorch_config(config):
    #TODO: error checking if we're generating fingerprints
    dataset_config = config["dataset"]
    if constants.CONFIG_FP_SCHEME in dataset_config:
        fp_scheme = dataset_config[constants.CONFIG_FP_SCHEME]
        if fp_scheme == "gmp":
            pass

            #TODO: validate mcsh parameters
        elif fp_scheme == "mcsh":
            #TODO: remove once we've switched over to gmp completely
            pass

        elif fp_scheme == "gaussian":
            pass

            #TODO: validate bp parameters
    
        else:
            raise RuntimeError("invalid fingerprint type: {}".format(fp_scheme))

def validate_eval_config(config):
    required_fields = [constants.CONFIG_EVAL_TYPE]

    for field in required_fields:
        if field not in config:
            raise RuntimeError("required field {} not in config".format(field))

#return directory containing current model checkpoint
#training_dir = training directory for a specific job
def get_checkpoint_dir(training_dir):
    #check if we're loading a model from checkpoint
    checkpoints_dir = training_dir / "checkpoints"

    #there should only be one checkpoint directory
    checkpoint_dirs = [path for path in checkpoints_dir.iterdir()]
    if len(checkpoint_dirs) != 1:
        raise RuntimeError("More than one directory present in the checkpoints directory")

    return checkpoint_dirs[0]