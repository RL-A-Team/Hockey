import pickle
import time
from argparse import ArgumentParser

import numpy as np
from laserhockey import hockey_env as h_env

parser = ArgumentParser()
parser.add_argument('--model', type=str, default='SAC/agent_sac.pkl', help='Path to pretrained model to load')
parser.add_argument('--games', type=int, default=10, help='Number of games to play')
parser.add_argument('--strong', action='store_true', help='Use strong basic opponent (else weak basic opponent)')
opts = parser.parse_args()

if __name__ == '__main__':
    agent = pickle.load(open(opts.model, 'rb'))
    agent.set_deterministic(True)

    env = h_env.HockeyEnv()

    if opts.strong:
        opponent = h_env.BasicOpponent(weak=False)
    else:
        opponent = h_env.BasicOpponent(weak=True)

    for game in range(opts.games):
        state, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        for step in range(100):
            a1 = agent.select_action(state)
            a2 = opponent.act(obs_agent2)
            next_state, _, done, _, info = env.step(np.hstack([a1, a2]))

            time.sleep(0.01)
            env.render()

            if done:
                break

            state = next_state
            obs_agent2 = env.obs_agent_two()

    env.close()