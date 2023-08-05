import os
import torch
import numpy as np
from Plots import Plots
from laserhockey import hockey_env as h_env
import sys
from DDPGAgent import DDPGAgent
from Parser import opts

if len(sys.argv) > 1:
    if isinstance(sys.argv[1], str):
        NAME = sys.argv[1]
    else:
        NAME = "test"
else:
    NAME = "test"

DIR = os.getcwd()
MODEL_DIR = os.getcwd() + '/weights/'
PRINT = True

# Training
def train(env_name):

    # initialize environment
    mode = opts.mode
    env = h_env.HockeyEnv(mode=mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space

    # initalize agent
    agent = DDPGAgent(state_dim, action_dim, opts.hidden_size_actor, opts.hidden_size_critic)
    if opts.model is not None:
        agent.load_model(opts.model)

    # get information from parser
    max_episodes = opts.max_episodes
    max_steps = opts.max_steps
    epsilon = opts.epsilon
    min_epsilon = opts.min_epsilon
    epsilon_decay = opts.epsilon_decay
    obs_noice = opts.obs_noice
    gamma = opts.gamma
    tau = opts.tau
    milestone = opts.milestone
    eval_interval = opts.eval_interval
    batch_size = opts.batch_size


    # initialize return information
    episode_reward = 0
    losses = np.zeros((max_episodes, 3))
    total_wins = 0
    total_losses = 0

    for episode in range(1, max_episodes+1):
        player2 = h_env.BasicOpponent(weak=True)
        if mode == 1:
            state,_ = env.reset(one_starting=True)  # shooting, player1 start
        elif mode == 2:
            state,_ = env.reset(one_starting=False)  # defending, player2 start
        else:
            state,_ = env.reset()  # nomral, change who starts

        episode_loss_actor = 0
        episode_loss_critic = 0
        reward_per_episode = 0
        puk_touched_counter = 1

        epsilon = max(epsilon_decay * epsilon, min_epsilon)

        for step in range(1, max_steps+1):
            obs_player2 = env.obs_agent_two()
            action1 = agent.get_action(state, epsilon = epsilon,
                                       obs_noice = obs_noice, evaluate= False, action_space=action_space)

            if (mode == 1):  # TRAIN_SHOOTING
                action2 = [0, 0, 0, 0]
            else:
                action2 = player2.act(obs_player2)

            next_state, reward, done, _, info = env.step(np.hstack([action1, action2]))
            puk_touched_counter += info["reward_touch_puck"]
            reward = 3 * reward + info["reward_closeness_to_puck"] + 3 * (
                        info["reward_puck_direction"] + 5 / puk_touched_counter * info["reward_touch_puck"])
            episode_reward += reward
            reward_per_episode += reward

            agent.store_experience(state, action1, next_state, reward, done)
            temp_loss = agent.train(batch_size, gamma, tau)
            state = next_state

            if temp_loss != []:
                episode_loss_actor += temp_loss[0]
                episode_loss_critic += temp_loss[1]

            if done:
                break
        if info['winner'] == 1:
            total_wins +=1
        elif info['winner'] == -1:
            total_losses += 1

        if (episode % eval_interval == 0 and PRINT):
            print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward/eval_interval:.2f}")
            episode_reward = 0

        if (step != 0):
            losses[episode-1] = [episode_loss_actor / step, episode_loss_critic / step, reward_per_episode]
        else:
            losses[episode-1] = [0 , 0, episode_reward]

        if (episode % milestone == 0):
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{NAME}_{env_name}-actor-m{mode}-h_size{opts.hidden_size_actor}-e{episode}.pth")
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{NAME}_{env_name}-critic-m{mode}-h_size{opts.hidden_size_critic}-e{episode}.pth")
    return {"reward": losses[:, 2], "actor_loss": losses[:, 0], "critic_loss": losses[:, 1],
            "stats": [["wins", "losses", "draws"], [total_wins, total_losses, max_episodes-(total_wins+total_losses)]]}


if __name__ == "__main__":

    info_dict = train("Hockey")
    plots = Plots(DIR)
    plots.save_results(info_dict)
    plots.plot_losses(info_dict["actor_loss"], title=f'actor_loss')
    plots.plot_losses(info_dict["critic_loss"], title=f'critic_loss')
    plots.plot_reward(info_dict["reward"], title=f'reward')
