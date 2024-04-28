import argparse
import os
import yaml
import subprocess

# Set up argument parser
parser = argparse.ArgumentParser(description='Train and validate model')
parser.add_argument('experiment_name', help='Name of the folder')

args = parser.parse_args()

experiment_name = args.experiment_name

model_path = os.path.join("/dtu/blackhole/08/193832", experiment_name)
print(f"The model location {model_path}")

config_file = os.path.join(model_path, "wandb/latest-run/files/config.yaml")
print(config_file)

def remove_keys(d):
    if isinstance(d, dict):
        if 'desc' in d:
            del d['desc']
        if 'value' in d:
            d = d['value']
            return d  # return immediately if 'd' is replaced
        for key in list(d.keys()):
            d[key] = remove_keys(d[key])
    elif isinstance(d, list):
        for i in range(len(d)):
            d[i] = remove_keys(d[i])
    return d


with open(config_file, 'r') as f:
    wandb_config = yaml.safe_load(f)

hydra_config = remove_keys(wandb_config)

if '_wandb' in hydra_config:
    del hydra_config['_wandb']

with open(os.path.join("./configs/model", experiment_name + ".yaml"), 'w') as f:
    yaml.safe_dump(hydra_config['model'], f)

#python your_script.py +hydra.run.dir=./new_config_dir +hydra.job.config_name=new_config_file
cmd = (f"python ./tools/train_val.py +model={experiment_name} model_name={experiment_name} evaluate_only=True")


print(cmd)
subprocess.run(cmd, shell=True)