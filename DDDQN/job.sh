#!/bin/bash
#SBATCH --job-name=HockeyDDDQN
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1    	  # optionally type and number of gpus
#SBATCH --time=00:15:00           # Runtime in D-HH:MM
#SBATCH --output=indiras_logs/outfiles/job_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=indiras_logs/errors/job_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=indira.echsler@student.uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

source $HOME/.bashrc

# insert your commands here
singularity exec --nv /home/stud54/Hockey/tcml_singularity_rl_lecture/rl_lecture python3 /home/stud54/Hockey/DDDQN/train.py
#singularity exec --nv /home/stud54/Hockey/tcml_singularity_rl_lecture/rl_lecture python3 /home/stud54/Hockey/DDDQN/train_factors.py
