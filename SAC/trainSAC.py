import os
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
parser.add_argument('--episodes', type=int, default=1500, help='Number of episodes to train')
parser.add_argument('--steps', type=int, default=500, help='Number of maximal steps per episode')
parser.add_argument('--mode', type=str, default='d', help='Training mode: d - defense, s - shooting, n - normal')
parser.add_argument('--model', type=str, default=None, help='Path to pretrained model to load')
parser.add_argument('--deterministic', action='store_true', help='Choose deterministic action (for evaluation)')
parser.add_argument('--reward', type=int, default=0, help='Code of reward to use')
parser.add_argument('--randomopponentdir', type=str, default=None, help='Directory to pick an random agent as opponent')

# SAC hyperparameter
parser.add_argument('--autotune', action='store_true', help='Autotune the entropy value')
parser.add_argument('--prb', action='store_true', help='Use Prioritized Replay Buffer')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the SAC model')
parser.add_argument('--tau', type=float, default=5e-3, help='Tau value for the SAC model')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('--batchsize', type=float, default=256, help='Batch size')
parser.add_argument('--loss', type=str, default='l1', help='Loss of the Critic Network (either l1 or l2)')
parser.add_argument('--gradientsteps', type=int, default=16, help='Gradient update steps after each rollout')

opts = parser.parse_args()


def print_opts(opts):
    """ Prints the defined options """

    print('--------------------------------------')
    print('---------TRAINING PARAMETER-----------')
    print(f'Training mode: {opts.mode}')
    print(f'Reward: {opts.reward}')
    print(f'Epsiodes: {opts.episodes}')
    print(f'Steps: {opts.steps}')
    print('')
    print(f'Load model: {opts.model}')
    print('')
    print(f'Alpha: {opts.alpha}')
    print(f'Tau: {opts.tau}')
    print(f'Learning rate: {opts.lr}')
    print(f'Discount factor: {opts.discount}')
    print(f'Batch size: {opts.batchsize}')
    print(f'Autotune: {opts.autotune}')
    print(f'Prioritized Replay Buffer: {opts.prb}')
    print(f'Loss: {opts.loss}')
    print(f'Gradient steps: {opts.gradientsteps}')
    print('--------------------------------------')
    print('')


def evaluate(agent, env, render):
    """ Evaluate the agent 25 times against the weak basic opponent

    :param agent: SAGAgent
    :param env: HockeyEnv
    :param render: Boolean
    :return: percent_win: float, percent_lose: float
    """
    agent.set_deterministic(True)

    eval_win = []
    eval_lose = []

    for eval_step in range(25):
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        # always evaluate against basic weak opponent (for better comparison)
        opponent = h_env.BasicOpponent(weak=True)

        for t in range(500):
            a1 = agent.select_action(state).detach().numpy()[0]

            a2 = opponent.act(obs_agent2)

            next_state, raw_reward, done, _, info = env.step(np.hstack([a1, a2]))

            state = next_state
            obs_agent2 = env.obs_agent_two()

            if render:
                time.sleep(0.01)
                env.render()

            if done:
                eval_win.append(1 if env.winner == 1 else 0)
                eval_lose.append(1 if env.winner == -1 else 0)
                break

    percent_win = eval_win.count(1) / len(eval_win)
    percent_lose = eval_lose.count(1) / len(eval_lose)
    agent.set_deterministic(False)

    return percent_win, percent_lose


if __name__ == '__main__':
    st = time.time()

    print_opts(opts)

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

    # load already trained model if desired, else initialize new agent
    if opts.model is not None:
        agent = pickle.load(open(opts.model, 'rb'))
    else:
        agent = SACAgent(state_dim=env.observation_space.shape, action_dim=env.action_space, alpha=opts.alpha,
                         tau=opts.tau, lr=opts.lr, discount=opts.discount, batch_size=opts.batchsize,
                         autotune=opts.autotune, loss=opts.loss, deterministic_action=opts.deterministic,
                         prio_replay_buffer=opts.prb)

    mean_rewards = []
    critic1_losses = []
    critic2_losses = []
    actor_losses = []
    alpha_losses = []
    stats_win = []
    stats_lose = []
    mean_win = []
    mean_lose = []
    eval_percent_win = []
    eval_percent_lose = []

    for episode in range(opts.episodes):
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        # select random opponent if desired, else use basic weak opponent
        if opts.randomopponentdir is not None:
            opponents = os.listdir(opts.randomopponentdir)
            opponents.append('weak')
            opponents.append('strong')
            opponents.append('strong')
            opponents.append('strong')
            opponents.append('strong')
            opponents.append('strong')
            opponents.append('strong')
            opponents.append('strong')
            choice = np.random.choice(opponents)

            if choice == 'weak':
                opponent = h_env.BasicOpponent(weak=True)
            elif choice == 'strong':
                opponent = h_env.BasicOpponent(weak=False)
            else:
                opponent = pickle.load(open(f'{opts.randomopponentdir}/{choice}', 'rb'))

            print('--------------------------------------')
            print(f'Random chosen opponent: {choice}')
            print('--------------------------------------')
            print('')
        else:
            opponent = h_env.BasicOpponent(weak=True)

        episode_rewards = []
        episode_win = []
        episode_lose = []
        last_touched_step = None
        first_touched_step = None

        touched = 0
        first_time_touch = 1

        for step in range(opts.steps):
            a1 = agent.select_action(state).detach().numpy()[0]

            if hasattr(opponent, 'act'):
                a2 = opponent.act(obs_agent2)
            else:
                a2 = opponent.select_action(state).detach().numpy()[0]

            next_state, raw_reward, done, _, info = env.step(np.hstack([a1, a2]))

            # reward = r
            if last_touched_step is not None:
                last_touched_step += 1
                first_touched_step += 1
            if info['reward_touch_puck'] == 1:
                last_touched_step = 0
                first_touched_step = 0

            if last_touched_step is not None and opts.reward in [1, 2, 3, 4, 5, 6]:
                # negative gompertz, use time since last touch
                decrease_touch = 1 - 0.99 * np.exp(-6 * np.exp(-0.3 * last_touched_step))
            elif first_touched_step is not None and opts.reward in [7, 8, 9]:
                # negative gompertz, use time since first touch
                decrease_touch = 1 - 0.99 * np.exp(-6 * np.exp(-0.3 * first_touched_step))
            else:
                decrease_touch = 0

            if opts.reward == -1:
                touched = max(touched, info['reward_touch_puck'])

                reward = (
                        raw_reward
                        + 5 * info['reward_closeness_to_puck']
                        - (1 - touched) * 0.1
                        + touched * first_time_touch * 0.1 * step
                )
                first_time_touch = 1 - touched
            elif opts.reward == 0:
                reward = raw_reward
            elif opts.reward == 1:
                # still very close to middle line, but better
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + 5 * decrease_touch + \
                         5 * info["reward_puck_direction"]
            elif opts.reward == 2:
                # goes directly to middle line
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + decrease_touch + 5 * info["reward_puck_direction"]
            elif opts.reward == 3:
                # goes directly to middle line
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + decrease_touch + info["reward_puck_direction"]
            elif opts.reward == 4:
                # sometimes does not even touch the ball (but does not go to middle line)
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + decrease_touch + 2.5 * info["reward_puck_direction"]
            elif opts.reward == 5:
                # goes directly to middle line
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + 2.5 * decrease_touch + \
                         5 * info["reward_puck_direction"]
            elif opts.reward == 6:
                # may work with more training
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + 0.5 * decrease_touch
            elif opts.reward == 7:
                # may work with more training
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + decrease_touch
            elif opts.reward == 8:
                # may also work
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + 2.5 * decrease_touch
            elif opts.reward == 9:
                # may also work
                reward = raw_reward + 4 * info["reward_closeness_to_puck"] + 0.5 * decrease_touch

            episode_rewards.append(reward)
            episode_win.append(1 if info['winner'] == 1 else 0)
            episode_lose.append(1 if env.winner == -1 else 0)

            agent.store_transition((state, a1, reward, next_state, done))

            if render:
                time.sleep(0.01)
                env.render()

            if done:
                break

            state = next_state
            obs_agent2 = env.obs_agent_two()

        # gradient descent of actor and critic network, track losses
        for step in range(opts.gradientsteps):
            critic1_loss, critic2_loss, actor_loss, alpha_loss = agent.update()
            critic1_losses.append(critic1_loss)
            critic2_losses.append(critic2_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)

        mean_rewards.append(np.array(episode_rewards).mean())
        mean_win.append(np.array(episode_win).mean())
        mean_lose.append(np.array(episode_lose).mean())
        stats_win.append(1 if env.winner == 1 else 0)
        stats_lose.append(1 if env.winner == -1 else 0)

        # evaluate every 500 episodes in deterministic mode against basic weak opponent
        if episode % 500 == 0:
            percent_win, percent_lose = evaluate(agent, env, render)
            eval_percent_win.append(percent_win)
            eval_percent_lose.append(percent_lose)

    env.close()

    # Save all results and generate plots
    utils.save_evaluation_results(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose,
                                  mean_rewards, mean_win, mean_lose, eval_percent_win, eval_percent_lose, agent)

    # print the execution time
    et = time.time()
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
