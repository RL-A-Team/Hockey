import numpy as np
import torch
from laserhockey import hockey_env as h_env
from dddqn import DDDQNAgent
import time
import matplotlib.pyplot as plt

# hyperparameters
weak_opponent = True
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL

episodes = 4000

load_checkpoint = False
save_checkpoint = False
visualize = False

hidden_dim = [300, 300] # number of hidden layers of neural network
alpha = 0.1             # actor loss weight: higher -> more exploration
tau = 5e-3              # rate at which target networks are updated using soft updates
learning_rate = 1e-3    # step size in updating the neural network 
discount = 0.99         # importance of future rewards
batch_size = 256        # transitions per update step
epsilon = 1e-6          # probability of selecting random action instead of policy action
max_size = 100000       # maximum capacity of replay buffer

if __name__ == '__main__':

    start_time = time.time()

    # game modi: TRAIN_DEFENSE, TRAIN_SHOOTING, NORMAL
    env = h_env.HockeyEnv(mode=game_mode)
    
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


    # load saved agent state from file
    if (load_checkpoint):
        checkpoint = torch.load('saved_agent.pth')
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])

    episode_counter = 0
    total_step_counter = 0
    grad_updates = 0
    new_op_grad = []

    # save metrics
    total_wins = 0
    total_reward = 0

    data_point_distance = 100
    data_point_episode_counter = 0
    data_point_sum = 0
    data_points = []

# main training loop
    while episode_counter < episodes:
        # reset environment, get initial state
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        # initialize opponent
        opponent = h_env.BasicOpponent(weak=weak_opponent)

        done = False
        episode_reward = 0
        first_touch = 1

        for step in range(1000):
            if (not done):
                # agent action
                a1 = agent.select_action(state).detach().numpy()[0]
                # opponent action
                a2 = opponent.act(obs_agent2)

                # take a step in the environment
                obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
                
                winner = info['winner']
                closeness_puck = info['reward_closeness_to_puck']
                touch_puck = info['reward_touch_puck']
                puck_direction = info['reward_puck_direction']

                if (touch_puck == 1):
                    first_touch = 0
                
                # compute a reward --- --- --- --- --- ---
                reward = reward

                factor = [1, 10, 100, 1]  # go to puck!
                #factor = [10, 1, 1, 10]   # shoot towards goal!
                #factor = [10, 5, 1, 1]   # go to puck, shoot goals!
                #reward = factor[0]*winner + factor[1]*closeness_puck + factor[2]*touch_puck + factor[3]*100*puck_direction
                
                #reward = 10*winner + 50*closeness_puck - (1-touch_puck) + (touch_puck*first_touch*step) + 100*puck_direction

                # --- --- --- --- --- --- --- --- --- --- ---

                # sum up total reward of episodes
                total_wins += winner if winner == 1 else 0
                data_point_sum += winner if winner == 1 else 0
                total_reward += reward

                agent.store_transition((state, a1, reward, obs, done))

                # visualization, not needed
                if (visualize):
                    time.sleep(0.01)
                    env.render()

                # update current state
                state = obs
                obs_agent2 = env.obs_agent_two()
                total_step_counter += 1

        data_point_episode_counter += 1

        if (data_point_episode_counter == 100):
            data_points.append(data_point_sum)

            data_point_episode_counter = 0
            data_point_sum = 0

        # update actor and critic networks
        critic1_loss, critic2_loss, actor_loss = agent.update()

        episode_counter += 1

        print(f'Episode {episode_counter}: Winner {env.winner}')

    if (save_checkpoint):
        # save the agent's two critics to file
        torch.save({
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        }, 'saved_agent.pth')
    

    # close environment
    env.close()
    print(f'Wins: {(total_wins/episodes)*100} %')
    print(f'Total reward {total_reward}')
    print(f'Reward per round {total_reward/episodes}')

    print()
    print(data_points)

    print()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Program execution took {execution_time:.4f} seconds.")
