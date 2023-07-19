import numpy as np
from laserhockey import hockey_env as h_env
from matplotlib import pyplot as plt

from sac import SACAgent
import time

if __name__ == '__main__':

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE)

    agent = SACAgent(state_dim=env.observation_space.shape, action_dim=env.action_space)

    episode_counter = 1
    total_step_counter = 0
    grad_updates = 0
    new_op_grad = []

    total_reward = 0

    critic1_losses = []
    critic2_losses = []
    actor_losses = []

    while episode_counter <= 5000:
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        opponent = h_env.BasicOpponent(weak=True)


        for step in range(250):
            a1 = agent.select_action(state).detach().numpy()[0]
            a2 = opponent.act(obs_agent2)

            ns, r, d, _, info = env.step(np.hstack([a1, a2]))

            total_reward += r

            agent.store_transition((state, a1, r, ns, d))

            time.sleep(0.01)
            #env.render()

            if d:
                break

            state = ns
            obs_agent2 = env.obs_agent_two()
            total_step_counter += 1

        critic1_loss, critic2_loss, actor_loss = agent.update()
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
        actor_losses.append(actor_loss)

        episode_counter += 1

        print(f'Epsiode {episode_counter}: Winner {env.winner}')

    plt.plot(np.arange(len(critic1_losses)), critic1_losses, label='Critic 1')
    plt.plot(np.arange(len(critic2_losses)), critic2_losses, label='Critic 2')
    plt.plot(np.arange(len(actor_losses)), actor_losses, label='Actor')
    plt.legend()
    plt.show()

    env.close()
    print(f'Total reward {total_reward}')