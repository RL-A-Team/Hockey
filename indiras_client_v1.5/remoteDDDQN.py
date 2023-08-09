from client import Client
from remoteControllerInterface import RemoteControllerInterface
from dddqn import DDDQNAgent
from laserhockey import hockey_env as h_env
import torch

agent_file = 'reward_f1_ft.pth'

class RemoteDDDQNAgent(DDDQNAgent, RemoteControllerInterface):

    def __init__(self, loaded, **kwargs):
        self.loaded = loaded
        RemoteControllerInterface.__init__(self, identifier='DDDQN')

    def remote_act(self, state):
        return self.loaded.select_action(state)

if __name__ == '__main__':

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv_BasicOpponent.NORMAL)
    
    hidden_dim = [300, 300] # number of hidden layers of neural network
    alpha = 0.1             # actor loss weight: higher -> more exploration
    tau = 5e-3              # rate at which target networks are updated using soft updates
    learning_rate = 1e-3    # step size in updating the neural network 
    discount = 0.96         # importance of future rewards
    batch_size = 256        # transitions per update step
    epsilon = 1e-4          # probability of selecting random action instead of policy action
    max_size = 1000000      # maximum capacity of replay buffer
    
    # initialize agent with state and action dimensions
    agent = DDDQNAgent(state_dim = env.observation_space.shape, 
                       action_dim = env.action_space, 
                       n_actions = 4, 
                       hidden_dim = hidden_dim, 
                       alpha = alpha, 
                       tau = tau, 
                       lr = learning_rate,
                       discount = discount, 
                       batch_size = batch_size,
                       epsilon = epsilon,
                       max_size = max_size)
    
    # load agent
    checkpoint = torch.load(agent_file) 
    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    agent.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
    agent.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1_state_dict'])
    agent.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    
    controller = RemoteDDDQNAgent(loaded=agent)
    
    # Play n (None for an infinite amount) games and quit
    client = Client(username='A Team',
                    password= 'too5xeit3T', #'too5xeit3T',
                    controller=controller,
                    output_path='logs', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
