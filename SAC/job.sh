#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:1    # optionally type and number of gpus
#SBATCH --mem=100G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/job_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/job_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=kristina.lietz@student.uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

source $HOME/.bashrc

# insert your commands here
#eval "$(micromamba shell hook --shell=bash)"
micromamba activate social-rl
pip install git+https://github.com/martius-lab/laser-hockey-env.git
srun python3 src/trainSAC.py 
micromamba deactivate

