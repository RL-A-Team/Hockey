from argparse import ArgumentParser
parser = ArgumentParser()
### slurm information
parser.add_argument('--job_id', default='test', help='slurm job idea')
parser.add_argument('--experiment', default='', help='type of experiment running')


parser.add_argument('--render', type=bool, default = False,
                    help='Render the training process (significantly increases running time)')
parser.add_argument('--max_episodes', type=int, default=10, help='Number of episodes to train')
parser.add_argument('--max_steps', type=int, default=250, help='Number of maximal steps per episode')
parser.add_argument('--mode', type=int, default=0, help='Training mode: 2 - defense, 1 - shooting, 0 - normal')
parser.add_argument('--model', type=str, default=None, help='Path to pretrained model to load')
parser.add_argument('--milestone', type=int, default=10, help='interval in which model is saved')
parser.add_argument('--eval_interval', type=int, default=5, help='interval to evaluate')

## DDPG hyperparameter

# exploration:
parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon value for exploration')
parser.add_argument('--epsilon_decay', type=float, default=0.997, help='(1-decay) of epsilon per timestep')
parser.add_argument('--min_epsilon', type=float, default=0.02, help='min value for epsilon')
parser.add_argument('--obs_noice', type=float, default=0.1, help='variance of noice added action for exploration')

# DDPG algoritm
parser.add_argument('--tau', type=float, default=5e-3, help='Tau value for the SAC model')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

## actor and critic models
parser.add_argument('--lr_actor', type=float, default=1e-3, help='Learning rate actor')
parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate critic')
parser.add_argument('--hidden_size_actor', type=int, default=128, help='hidden size actor')
parser.add_argument('--hidden_size_critic', type=int, default=128, help='hidden size critic')
parser.add_argument('--batch_size', type=float, default=256, help='Batch size')

opts = parser.parse_args()