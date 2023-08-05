from DDPGAgent import DDPGAgent
from laserhockey import hockey_env as h_env
import os
import numpy as np
import time

class Evaluater():
    def __init__(self, step_size, visualize = True, path=None):
        self.step_size = step_size
        if path == None:
            self.path = os.getcwd() + '/weights/'
        else:
            self.path = path
        self.visualize = visualize

    def evaluate_one_model(self, max_episodes, model, hidden_size, mode, opponment = None):
        env = h_env.HockeyEnv(mode=mode)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # inititalize player 1
        agent = DDPGAgent(state_dim, action_dim, hidden_size_actor=hidden_size)
        agent.load_model(self.path + model)

        # initialize player 2
        if opponment is None:
            player2 = h_env.BasicOpponent(weak=True)
        else:
            player2 = opponment
        rewards = []
        total_wins = 0
        total_losses = 0

        for episode in range(max_episodes):
            if mode == 1:
                state, _ = env.reset(one_starting=True)  # shooting, player1 start
            elif mode == 2:
                state, _ = env.reset(one_starting=False)  # defending, player2 start
            else:
                state, _ = env.reset()  # nomral, change who starts
            total_reward = 0

            for step in range(self.step_size):
                if self.visualize:
                    time.sleep(0.01)
                    env.render("human")

                action1 = agent.get_action(state, epsilon=0, evaluate=True)
                obs_player2 = env.obs_agent_two()
                if (mode == 1):  # TRAIN_SHOOTING
                    action2 = [0, 0, 0, 0]
                else:
                    action2 = player2.act(obs_player2)
                next_state, reward, done, _, info = env.step(np.hstack([action1, action2]))
                total_reward += reward
                state = next_state

                if done:
                    break
            if info["winner"] == 1:
                total_wins += 1
            elif info["winner"] == -1:
                total_losses += 1
            rewards.append(total_reward)


            print(f'epsidoe {episode} ended after {step} steps with a total reward of {total_reward}')
        return {"eval_reward": rewards,
                "eval_stats": [["wins", "losses", "draws"], [total_wins, total_losses, max_episodes-(total_wins+total_losses)]]}