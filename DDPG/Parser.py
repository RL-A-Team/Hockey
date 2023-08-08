from argparse import ArgumentParser
parser = ArgumentParser()
### slurm information
parser.add_argument('--job_id', default='test', help='slurm job idea')
parser.add_argument('--experiment', default='', help='type of experiment running')
parser.add_argument('--seed', default=None, help="set random seed")


parser.add_argument('--render', type=bool, default = False,
                    help='Render the training process (significantly increases running time)')
parser.add_argument('--max_episodes', type=int, default=10, help='Number of episodes to train')
parser.add_argument('--max_steps', type=int, default=250, help='Number of maximal steps per episode')
parser.add_argument('--mode', type=int, default=0, help='Training mode: 2 - defense, 1 - shooting, 0 - normal')
parser.add_argument('--model', type=str, default=None, help='Path to pretrained model to load')
parser.add_argument('--milestone', type=int, default=1000, help='interval in which model is saved')
parser.add_argument('--eval_interval', type=int, default=20, help='interval to evaluate')
parser.add_argument('--update_target_every', type=int, default=100, help='update target every x episdoes')
parser.add_argument('--iter_train', type=int, default=32, help='amount of batches to train on per episode')


## DDPG hyperparameter

# exploration:
parser.add_argument('--epsilon', type=float, default=0.3, help='epsilon value for exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.9997, help='(1-decay) of epsilon per timestep')
parser.add_argument('--min_epsilon', type=float, default=0.1, help='min value for epsilon')

# DDPG algoritm
parser.add_argument('--tau', type=float, default= 5e-3, help='Tau value for the SAC model')
parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')

## actor and critic models
parser.add_argument('--lr_actor', type=float, default=0.0001, help='Learning rate actor')
parser.add_argument('--lr_critic', type=float, default=0.0001, help='Learning rate critic')
parser.add_argument('--hidden_size_actor', type=int, default=[128, 128], help='hidden size actor')
parser.add_argument('--hidden_size_critic', type=int, default=[128,128, 64], help='hidden size critic')
parser.add_argument('--batch_size', type=float, default=256, help='Batch size')

opts = parser.parse_args()