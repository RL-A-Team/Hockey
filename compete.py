import numpy as np
import torch
import time

import pickle
import sys
sys.path.append('SAC')

from laserhockey import hockey_env as h_env

from DDQN.ddqn import DDDQNAgent
from SAC.sac import SACAgent

# for saving gif :)
from PIL import Image
import imageio
step_interval = 2
output_gif_path = 'compete.gif'
frames = []

agent_file_1 = 'SAC/agent_sac.pkl'      # .pkl file
agent_file_2 = 'DDQN/agent_ddqn.pth'    # .pth file


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#                               hyperparameters                               #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                                                                                        #                                                                                       
episodes = 3                                                                          #
visualize = True                                                                        #
                                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                                                                                                

if __name__ == '__main__':

    start_time = time.time()

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv_BasicOpponent.NORMAL)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#                                 load agent 1                                #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                                                                        #    
    agent_1 = SACAgent(state_dim = env.observation_space.shape,         #                              
                         action_dim = env.action_space)                 #
                                                                        #
    agent_1 = pickle.load(open(agent_file_1, 'rb'))                     #
                                                                        #    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#                                 load agent 2                                #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                                                                                                        #    
    hidden_dim = [300, 300] # number of hidden layers of neural network                                 #
    alpha = 0.1             # actor loss weight: higher -> more exploration                             #
    tau = 5e-3              # rate at which target networks are updated using soft updates              #
    learning_rate = 1e-3    # step size in updating the neural network                                  #
    discount = 0.96         # importance of future rewards                                              #   
    batch_size = 256        # transitions per update step                                               #
    epsilon = 1e-4          # probability of selecting random action instead of policy action           #
    max_size = 1000000      # maximum capacity of replay buffer                                         #
                                                                                                        #
    # initialize agent with state and action dimensions                                                 #   
    agent_2 = DDDQNAgent(state_dim = env.observation_space.shape,                                       #
                         action_dim = env.action_space,                                                 #   
                         n_actions = 4,                                                                 #
                         hidden_dim = hidden_dim,                                                       #
                         alpha = alpha,                                                                 #   
                         tau = tau,                                                                     #
                         lr = learning_rate,                                                            #
                         discount = discount,                                                           #
                         batch_size = batch_size,                                                       #
                         epsilon = epsilon,                                                             #
                         max_size = max_size)                                                           #
                                                                                                        #
    # load agent from the agent_file                                                                    #
    checkpoint = torch.load(agent_file_2)                                                               #
    agent_2.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])                                 #
    agent_2.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])                                 #
    agent_2.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])                   #
    agent_2.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])                   #
    agent_2.actor.load_state_dict(checkpoint['actor_state_dict'])                                       #
    agent_2.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1_state_dict'])             #
    agent_2.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2_state_dict'])             #
    agent_2.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])                   #
                                                                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    episode_counter = 0

    # save metrics
    total_wins = 0
    total_losses = 0
    total_draws = 0

# main loop
    while episode_counter < episodes:

        # reset environment, get initial state
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        done = False
        winner = 0

        for step in range(1000):
            if (not done):
                # agent 1 action
                a1 = agent_1.select_action(state)

                # agent 2 action
                a2 = agent_2.select_action(obs_agent2).detach().numpy()[0]

                # take a step in the environment
                obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
                
                winner = info['winner']

                # visualization
                if (visualize):
                    time.sleep(0.01)

                    env.render()

                    # collect gif frames
                    if (step + 1) % step_interval == 0:
                        frame = env.render(mode='rgb_array')
                        pil_frame = Image.fromarray(frame)
                        frames.append(pil_frame)

                # update current state
                state = obs
                obs_agent2 = env.obs_agent_two()

        # sum up results
        total_wins += 1 if winner == 1 else 0
        total_losses += 1 if winner == -1 else 0

        episode_counter += 1

        print(f'Episode {episode_counter}: Winner {env.winner}')

    # close environment
    env.close()

    print()
    print(f'left (SAC wins): {total_wins}')
    print(f'right (DDQN) wins: {total_losses}')
    print(f'Draws: {episodes - (total_wins + total_losses)}')

    print()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program execution took {execution_time:.4f} seconds.")

    # save gif
    imageio.mimsave(output_gif_path, frames, duration=0.025)
