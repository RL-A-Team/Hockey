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

if __name__ == '__main__':
    st = time.time()

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

        opponent = h_env.BasicOpponent(weak=True)

        episode_rewards = []
        episode_win = []
        episode_lose = []
        last_touched_step = None

        for step in range(opts.steps):
            a1 = agent.select_action(state).detach().numpy()[0]
            a2 = opponent.act(obs_agent2)

            ns, r, d, _, info = env.step(np.hstack([a1, a2]))

            #reward = r
            if last_touched_step is not None:
                last_touched_step += 1
            if info['reward_touch_puck'] == 1:
                last_touched_step = 0

            if last_touched_step is not None:
                # negative gompertz
                decrease_touch = 1 - 0.99 * np.exp(-6 * np.exp(-0.3 * last_touched_step))

                # first reward
                #reward = r + 4*info["reward_closeness_to_puck"] + 5*decrease_touch + 5*info["reward_puck_direction"]

                # second reward
                reward = r + 4 * info["reward_closeness_to_puck"] + decrease_touch + 5 * info["reward_puck_direction"]
            else:
                decrease_touch = 0

            if opts.reward == 0:
                reward = r
            elif opts.reward == 1:
                reward = r + 4 * info["reward_closeness_to_puck"] + 5 * decrease_touch + \
                         5 * info["reward_puck_direction"]
            elif opts.reward == 2:
                reward = r + 4 * info["reward_closeness_to_puck"] + decrease_touch + 5 * info["reward_puck_direction"]
            elif opts.reward == 3:
                reward = r + 4 * info["reward_closeness_to_puck"] + decrease_touch + info["reward_puck_direction"]
            elif opts.reward == 4:
                reward = r + 4 * info["reward_closeness_to_puck"] + decrease_touch + 2.5 * info["reward_puck_direction"]
            elif opts.reward == 5:
                reward = r + 4 * info["reward_closeness_to_puck"] + 2.5 * decrease_touch + \
                         5 * info["reward_puck_direction"]

            episode_rewards.append(reward)
            episode_win.append(1 if info['winner'] == 1 else 0)
            episode_lose.append(1 if env.winner == -1 else 0)

            agent.store_transition((state, a1, reward, ns, d))

            if render:
                time.sleep(0.01)
                env.render()

            if d:
                break

            state = ns
            obs_agent2 = env.obs_agent_two()

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

        # evaluate every 500 episodes
        if episode % 500 == 0:
            agent.set_deterministic(True)

            eval_win = []
            eval_lose = []

            for eval_step in range(25):
                state, info = env.reset()
                obs_agent2 = env.obs_agent_two()

                opponent = h_env.BasicOpponent(weak=True)

                for t in range(500):
                    a1 = agent.select_action(state).detach().numpy()[0]
                    a2 = opponent.act(obs_agent2)

                    ns, r, d, _, info = env.step(np.hstack([a1, a2]))

                    state = ns
                    obs_agent2 = env.obs_agent_two()

                    if render:
                        time.sleep(0.01)
                        env.render()

                    if d:
                        eval_win.append(1 if env.winner == 1 else 0)
                        eval_lose.append(1 if env.winner == -1 else 0)
                        break

            eval_percent_win.append(eval_win.count(1)/len(eval_win))
            eval_percent_lose.append(eval_lose.count(1) / len(eval_lose))
            agent.set_deterministic(False)



        # print(f'Episode {episode+1}: Winner {env.winner}')

    env.close()

    utils.save_evaluation_results(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose,
                                  mean_rewards, mean_win, mean_lose, eval_percent_win, eval_percent_lose, agent, False)

    # print the execution time
    et = time.time()
    elapsed_time = et - st
    print("")
    print('Execution time:', round(elapsed_time / 60), 'minutes')
