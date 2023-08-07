import numpy as np
import torch
from laserhockey import hockey_env as h_env
from dddqn import DDDQNAgent

import time
import matplotlib.pyplot as plt


# parameters for manual configuration
weak_opponent = True
game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
#game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
episodes = 1000
use_checkpoint = True
visualize = False

factor = [5, 3, 2, 4]

if __name__ == '__main__':
    # game modi: TRAIN_DEFENSE, TRAIN_SHOOTING, NORMAL
    
    env = h_env.HockeyEnv(mode=game_mode)
    
    # initialize agent with state and action dimensions
    agent = DDDQNAgent(state_dim=env.observation_space.shape, action_dim=env.action_space)

    # load saved agent state from file
    if (use_checkpoint):
        checkpoint = torch.load('saved_agent.pth')
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])

    episode_counter = 0
    total_step_counter = 0
    grad_updates = 0
    new_op_grad = []

    total_wins = 0
    total_reward = 0

# main training loop
    while episode_counter < episodes:
        # reset environment, get initial state
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        # initialize opponent
        opponent = h_env.BasicOpponent(weak=weak_opponent)

        done = False
        episode_reward = 0

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
                total_wins += winner
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

    
    # save the agent's two critics to file
    torch.save({
    'critic_1_state_dict': agent.critic_1.state_dict(),
    'critic_2_state_dict': agent.critic_2.state_dict(),
}, 'saved_agent.pth')
    

    # close environment
    env.close()
    print(f'Total wins {total_wins}')
    print(f'Wins per round {total_wins/episodes}')
    print(f'Total reward {total_reward}')
    print(f'Reward per round {total_reward/episodes}')

    
