import pickle

import numpy as np
from laserhockey import hockey_env as h_env
from argparse import ArgumentParser
from sac import SACAgent
import time

import utils

parser = ArgumentParser()
parser.add_argument('--render', action='store_true',
                    help='Render the training process (significantly increases running time)')
parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
parser.add_argument('--steps', type=int, default=500, help='Number of maximal steps per episode')
parser.add_argument('--mode', type=str, default='d', help='Training mode: d - defense, s - shooting, n - normal')
parser.add_argument('--model', type=str, default=None, help='Path to pretrained model to load')

# SAC hyperparameter
parser.add_argument('--autotune', action='store_true', help='Autotune the entropy value')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the SAC model')
parser.add_argument('--tau', type=float, default=5e-3, help='Tau value for the SAC model')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('--batchsize', type=float, default=256, help='Batch size')

opts = parser.parse_args()

if __name__ == '__main__':

    if opts.mode == 'd' or opts.mode == 'defense':
        mode = h_env.HockeyEnv.TRAIN_DEFENSE
    elif opts.mode == 's' or opts.mode == 'shooting':
        mode = h_env.HockeyEnv.TRAIN_SHOOTING
    elif opts.mode == 'n' or opts.mode == 'normal':
        mode = h_env.HockeyEnv.NORMAL
    else:
        raise ValueError(f'Mode {opts.mode} not defined! Please define mode as d - defense, s - shooting or n - normal')

    env = h_env.HockeyEnv(mode=mode)

    render = opts.render

    if opts.model is not None:
        agent = pickle.load(open(opts.model, 'rb'))
    else:
        agent = SACAgent(state_dim=env.observation_space.shape, action_dim=env.action_space, alpha=opts.alpha,
                         tau=opts.tau, lr=opts.lr, discount=opts.discount, batch_size=opts.batchsize,
                         autotune=opts.autotune)

    critic1_losses = []
    critic2_losses = []
    actor_losses = []
    alpha_losses = []
    stats_win = []
    stats_lose = []

    for episode in range(opts.episodes):
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        opponent = h_env.BasicOpponent(weak=True)

        for step in range(opts.steps):
            a1 = agent.select_action(state).detach().numpy()[0]
            a2 = opponent.act(obs_agent2)

            ns, r, d, _, info = env.step(np.hstack([a1, a2]))

            reward = r + 10 * info['reward_closeness_to_puck'] + 10 * info['reward_puck_direction']

            agent.store_transition((state, a1, reward, ns, d))

            if render:
                time.sleep(0.01)
                env.render()

            if d:
                break

            state = ns
            obs_agent2 = env.obs_agent_two()

        critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.update()
        critic1_losses.append(critic1_loss)
        critic2_losses.append(critic2_loss)
        actor_losses.append(actor_loss)
        alpha_losses.append(alpha_loss)

        stats_win.append(1 if env.winner == 1 else 0)
        stats_lose.append(1 if env.winner == -1 else 0)

        print(f'Episode {episode+1}: Winner {env.winner}')

    env.close()

    utils.save_evaluation_results(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose,
                                  agent, False)
