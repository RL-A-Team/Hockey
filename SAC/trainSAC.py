import pickle

import numpy as np
from laserhockey import hockey_env as h_env

from sac import SACAgent
import time

import utils

if __name__ == '__main__':

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv_BasicOpponent.NORMAL)

    render = True

    #agent = SACAgent(state_dim=env.observation_space.shape, action_dim=env.action_space, autotune=True)
    agent = pickle.load(open('models/sac_model_20230728T224813.pkl', 'rb'))

    episode_counter = 1
    total_step_counter = 0
    grad_updates = 0
    new_op_grad = []

    critic1_losses = []
    critic2_losses = []
    actor_losses = []
    alpha_losses = []
    stats_win = []
    stats_lose = []

    while episode_counter <= 5000:
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        opponent = h_env.BasicOpponent(weak=True)


        for step in range(250):
            a1 = agent.select_action(state).detach().numpy()[0]
            a2 = opponent.act(obs_agent2)

            ns, r, d, _, info = env.step(np.hstack([a1, a2]))

            reward = r + 10*info['reward_closeness_to_puck'] + 10*info['reward_puck_direction']

            agent.store_transition((state, a1, reward, ns, d))

            if render:
                time.sleep(0.01)
                env.render()

            if d:
                break

            state = ns
            obs_agent2 = env.obs_agent_two()
            total_step_counter += 1

        critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.update()
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
        actor_losses.append(actor_loss)
        alpha_losses.append(alpha_loss)

        stats_win.append(1 if env.winner == 1 else 0)
        stats_lose.append(1 if env.winner == -1 else 0)

        episode_counter += 1

        print(f'Epsiode {episode_counter}: Winner {env.winner}')

    env.close()

    utils.save_evaluation_results(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose,
                                  agent, False)
