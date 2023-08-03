import numpy as np
import torch
from laserhockey import hockey_env as h_env
from dddqn import DDDQNAgent

import time

# parameters for manual configuration
weak = True
#game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
#game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
episodes = 1000

def get_device():
    # get appropriate device for torch tensors
    # use GPU if available, else CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # game modi: TRAIN_DEFENSE, TRAIN_SHOOTING, NORMAL
    
    env = h_env.HockeyEnv(mode=game_mode)
    
    # initialize agent with state and action dimensions
    agent = DDDQNAgent(state_dim=env.observation_space.shape, action_dim=env.action_space)

    # load saved agent state from file
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
        opponent = h_env.BasicOpponent(weak=weak)

        for step in range(100):
            # agent action
            a1 = agent.select_action(state).detach().cpu().numpy()[0]
            # opponent action
            a2 = opponent.act(obs_agent2)

            # take a step in the environment
            ns, r, d, _, info = env.step(np.hstack([a1, a2]))

            # sum up total reward of episodes
            total_reward += r

            agent.store_transition((state, a1, r, ns, d))

            # visualization, not needed
            #time.sleep(0.01)
            #env.render()

            # update current state
            state = ns
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

