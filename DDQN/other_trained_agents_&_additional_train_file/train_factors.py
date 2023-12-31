import numpy as np
import torch
from laserhockey import hockey_env as h_env
from ddqn import DDDQNAgent

import time
import sys

agent_file = 'saved_agent_uwu.pth'

##########
current_subtask = 0
subtasks = 25
##########

# hyperparameters
weak_opponent = True
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL

episodes = 1000

load_checkpoint = False
save_checkpoint = False
visualize = False

hidden_dim = [300, 300] # number of hidden layers of neural network
alpha = 0.1             # actor loss weight: higher -> more exploration
tau = 5e-3              # rate at which target networks are updated using soft updates
learning_rate = 1e-3    # step size in updating the neural network 
discount = 0.96         # importance of future rewards
batch_size = 256        # transitions per update step
epsilon = 1e-4          # probability of selecting random action instead of policy action
max_size = 1000000      # maximum capacity of replay buffer


factors = []
fac_rewards = []
fac_wins = []

for a in [1,10,100]:
    for b in [1,10,100]:
        for c in [1,10,100]:
            for d in [1,10,100]:
                factors.append([a, b, c, d])

# split factors to subtasks
sub_fac_len = len(factors) // subtasks

sub_fac = []

# split the original array into sub-arrays
for i in range(0, len(factors), sub_fac_len):
    sub_fac.append(factors[i:i+sub_fac_len])

# if there are any remaining elements, add them to the last sub-array
if len(factors) % sub_fac_len != 0:
    sub_fac[-1] += factors[-(len(factors) % sub_fac_len):]

factors = sub_fac[current_subtask]

count = 0

start_time = time.time()

for factor in factors:

    print(count)
    count += 1

    if __name__ == '__main__':

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
            checkpoint = torch.load(agent_file)
            
            agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            agent.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
            agent.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1_state_dict'])
            agent.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        episode_counter = 0
        total_step_counter = 0
        grad_updates = 0
        new_op_grad = []

        total_wins = 0
        total_losses = 0
        total_reward = 0
        

    # main training loop
        while episode_counter < episodes:
            # reset environment, get initial state
            state, info = env.reset()
            obs_agent2 = env.obs_agent_two()

            # initialize opponent
            opponent = h_env.BasicOpponent(weak=weak_opponent)

            done = False

            for step in range(1000):
                if (not done):
                    # agent action
                    a1 = agent.select_action(state).detach().numpy()[0]
                    # opponent action
                    a2 = opponent.act(obs_agent2)

                    # take a step in the environment
                    obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
                    
                    # compute a reward
                    winner = info['winner']
                    closeness_puck = info['reward_closeness_to_puck']
                    touch_puck = info['reward_touch_puck']
                    puck_direction = info['reward_puck_direction']*100
                    
                    reward = factor[0]*winner + factor[1]*closeness_puck + factor[2]*touch_puck + factor[3]*puck_direction
                    
                    # sum up total reward of episodes
                    total_wins += winner if winner == 1 else 0
                    total_losses += 1 if winner == -1 else 0
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

            # update actor and critic networks
            critic1_loss, critic2_loss, actor_loss = agent.update()

            episode_counter += 1

            print(f'Episode {episode_counter}: Winner {env.winner}')

        
        if (save_checkpoint):
            checkpoint = {
            'critic_1_state_dict': agent.critic_1.state_dict(),
            'critic_2_state_dict': agent.critic_2.state_dict(),
            'critic_target_1_state_dict': agent.critic_target_1.state_dict(),
            'critic_target_2_state_dict': agent.critic_target_2.state_dict(),
            'actor_state_dict': agent.actor.state_dict(),
            'critic_optimizer_1_state_dict': agent.critic_optimizer_1.state_dict(),
            'critic_optimizer_2_state_dict': agent.critic_optimizer_2.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            }
        
            torch.save(checkpoint, agent_file)
        

        # close environment
        env.close()
        print(f'Total reward {total_reward}')
        print(f'Reward per round {total_reward/episodes}')
        
        fac_rewards.append(total_reward)
        fac_wins.append(total_wins)

# factor winner for max reward
max_rew = max(fac_rewards)
max_rew_index = fac_rewards.index(max_rew)
print()
print('BEST REWARD RESULT OF SUBTASK ' + str(current_subtask) + ': ')
print('max_rew: ' + str(max_rew))
print('wins: ' + str(fac_wins[max_rew_index]))
print('best factors: ' + str(factors[max_rew_index]))

# factor winner for max wins
max_win = max(fac_wins)
max_win_index = fac_wins.index(max_win)
print()
print('BEST WINNER RESULT OF SUBTASK ' + str(current_subtask) + ': ')
print('reward: ' + str(fac_rewards[max_win_index]))
print('max_win: ' + str(fac_wins[max_win_index]))
print('best factors: ' + str(factors[max_win_index]))

print()
end_time = time.time()
execution_time = end_time - start_time
print(f"Program execution took {execution_time:.4f} seconds.")

print('copy version:')
print(str(current_subtask) + '(rew*)  ' + str(fac_wins[max_rew_index]) + '  ' + str(fac_rewards[max_rew_index]) + ' ' + str(factors[max_rew_index]))
print(str(current_subtask) + '(win*)  ' + str(fac_wins[max_win_index]) + '  ' + str(fac_rewards[max_win_index]) + ' ' + str(factors[max_win_index]))
