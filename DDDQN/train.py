import numpy as np
import torch
from laserhockey import hockey_env as h_env
from dddqn import DDDQNAgent

import time

# parameters for manual configuration
weak_opponent = True
game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
#game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
episodes = 100
use_checkpoint = False
visualize = False

factor = [5,1,1,3]

# # # # # # # # # # TODO kick
factors = []


fac_rewards = []

for a in range(1, 6):
    for b in range(1, 6):
        for c in range(1, 6):
            for d in range(1, 6):
                factors.append([a, b, c, d])

count = 0
for factor in factors:

    print(count)
    count += 1
    # # # # # # # # # # 

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

        total_reward = 0

    # main training loop
        while episode_counter <= episodes:
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
                    puck_direction = info['reward_puck_direction']
                    
                    # +1, because it looks nice and I am happy if the total reward looks good         
                    reward = factor[0]*winner + factor[1]*closeness_puck + factor[2]*touch_puck + factor[3]*puck_direction
                    
                    # sum up total reward of episodes
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
        print(f'Total reward {total_reward}')
        print(f'Reward per round {total_reward/episodes}')

        # # # # # # # # # # TODO kick
        fac_rewards.append(total_reward)


# factor winner
max_rew = max(fac_rewards)
max_rew_index = fac_rewards.index(max_rew)
print(max_rew)
print(max_rew_index)
print(factors[max_rew_index])

# # # # # # # # # # 