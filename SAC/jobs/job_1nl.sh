#!/bin/bash
#SBATCH --job-name=HockeySAC
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1    	# optionally type and number of gpus
#SBATCH --time=00:15:00            # Runtime in D-HH:MM
#SBATCH --output=logs/job_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/job_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=kristina.lietz@student.uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

source $HOME/.bashrc

# insert your commands here
singularity exec --nv /mnt/qb/work/ludwig/klietz10/Hockey/tcml_singularity_rl_lecture/rl_lecture python3 /mnt/qb/work/ludwig/klietz10/Hockey/SAC/trainSAC.py --episodes 1000 --autotune --prb --loss l2 --lr 1e-5 --reward 7 --model /mnt/qb/work/ludwig/klietz10/Hockey/models/sac_model_20230808T110807_18777.pkl --mode normal --randomopponentdir /home/stud54/Hockey/models