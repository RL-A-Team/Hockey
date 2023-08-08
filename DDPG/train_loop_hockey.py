<<<<<<< HEAD
import os
import torch
import numpy as np
from Plots import Plots
from laserhockey import hockey_env as h_env
import sys
from DDPGAgent import DDPGAgent
from Parser import opts
#from Evaluater import Evaluater

DIR = os.getcwd()
MODEL_DIR = os.getcwd() + '/weights/'
PRINT = True

# Training
def train(env_name, TEST):

    # get information from parser
    NAME = opts.job_id
    max_episodes = opts.max_episodes
    max_steps = opts.max_steps
    epsilon = opts.epsilon
    min_epsilon = opts.min_epsilon
    epsilon_decay = opts.epsilon_decay
    gamma = opts.gamma
    tau = opts.tau
    milestone = opts.milestone
    eval_interval = opts.eval_interval
    batch_size = opts.batch_size
    update_target_every = opts.update_target_every
    iter_train = opts.iter_train

    # initialize environment
    weak_opponent = True

    game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    if opts.mode == 1:
        game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        game_mode == h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING

    env = h_env.HockeyEnv(mode=game_mode)
    action_dim = env.action_space.shape[0]

    # initalize agent
    agent =  DDPGAgent(env.observation_space, env.action_space, action_dim, tau=tau, gamma=gamma,
                     hidden_size_actor=opts.hidden_size_actor, lr_actor = opts.lr_actor,
                     hidden_size_critic=opts.hidden_size_critic, lr_critic=opts.lr_critic,
                     update_target_every = opts.update_target_every)
    if opts.model is not None:
        agent.load_model(opts.model)

    # initialize return information

    rewards_res = []
    actor_loss_res = []
    critic_loss_res = []
    lengths = []
    won = []
    lost= []
    draws = []
    total_wins = 0
    total_losses = 0

    for episode in range(1, max_episodes+1):
        player2 = h_env.BasicOpponent(weak=True)
        state, _ = env.reset()

        puk_touched_counter = 1
        total_reward = 0

        agent.reset()
        epsilon = max(epsilon_decay * epsilon, min_epsilon)
        done = False

        for step in range(1, max_steps+1):
            obs_player2 = env.obs_agent_two()
            action1 = agent.act(state, eps = epsilon)
            action2 = player2.act(obs_player2)

            next_state, reward, done, trunc, info = env.step(np.hstack([action1, action2]))
            puk_touched_counter += info["reward_touch_puck"]
            reward = 3 * reward + info["reward_closeness_to_puck"] + 3 * (
                        info["reward_puck_direction"] + (5 / puk_touched_counter) * info["reward_touch_puck"])
            total_reward += reward
            agent.store_transition(state, action1, next_state, reward, done)
            state = next_state

            if done or trunc:
                break

        if info['winner'] == 1:
            total_wins +=1
            won.append([1])
            lost.append([0])
            draws.append([0])
        elif info['winner'] == -1:
            total_losses += 1
            won.append([0])
            lost.append([1])
            draws.append([0])
        else:
            won.append([0])
            lost.append([0])
            draws.append([1])
        actor_loss, critic_loss = agent.train(batch_size)
        actor_loss_res.append(actor_loss)
        critic_loss_res.append(critic_loss)
        rewards_res.append(total_reward)
        lengths.append(step)

        #logging
        if episode % eval_interval == 0:
            avg_reward = np.mean(rewards_res[-eval_interval:])
            avg_length = int(np.mean(lengths[-eval_interval:]))
            avg_wins = np.mean(won[-eval_interval:])
            avg_lost = np.mean(lost[-eval_interval:])
            avg_draws = np.mean(draws[-eval_interval:])

            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
            print('won {} \t lost: {} \t draw: {}'.format(avg_wins, avg_lost, avg_draws))

        if (episode % milestone == 0):
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{TEST}/{NAME}_{env_name}_test{opts.tau}_actor_m{opts.mode}_"
                                                 f"h-size{opts.hidden_size_actor}_lr{opts.lr_actor}_e{episode}.pth")
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{TEST}/{NAME}_{env_name}_test{opts.tau}_critic_m{opts.mode}_"
                                                 f"h-size{opts.hidden_size_critic}_lr{opts.lr_critic}_e{episode}.pth")
    return {"reward": rewards_res, "actor_loss": actor_loss_res, "critic_loss": critic_loss_res, "wins": won,
            "lost": lost, "draws": draws,
            "stats": [["wins", "losses", "draws"], [total_wins, total_losses, max_episodes-(total_wins+total_losses)]]}

if __name__ == "__main__":
    NAME = opts.job_id
    TEST = opts.iter_train
    train_result = train("Hockey", TEST)
    plots_train = Plots(DIR+"/results/train_results/"+str(TEST))
    plots_train.save_results(train_result)
    running_mean = 20
    plots_train.plot_res(train_result["actor_loss"],  running_mean = running_mean, model=NAME, title=f'train_actor_loss')
    plots_train.plot_res(train_result["critic_loss"], running_mean = running_mean,model=NAME, title=f'train_critic_loss')
    plots_train.plot_res(train_result["reward"], running_mean = running_mean,model=NAME, title=f'train_reward')
    plots_train.plot_res(train_result["wins"], running_mean = running_mean,model=NAME, title=f'wins')
    plots_train.plot_res(train_result["lost"], running_mean = running_mean,model=NAME, title=f'losses')
    plots_train.plot_res(train_result["draws"], running_mean = running_mean,model=NAME, title=f'draws')






    print(f'########## Hyperparameter: ####################'
          f'max_episodes : {opts.max_episodes} \n' 
          f'mode: {opts.mode} \n'
          f'epsilon: {opts.epsilon} \n'
          f'epsilon_decay: {opts.epsilon_decay} \n'
          f'tau: {opts.tau}\n'
          f'gamma: {opts.gamma}\n'
          f'lr_actor: {opts.lr_actor}\n'
          f'hidden_size_actor: {opts.hidden_size_actor}\n'
          f'lr_critic: {opts.lr_critic}\n'
          f'hidden_size_critic: {opts.hidden_size_critic}\n'
          f'batch_size: {opts.batch_size}\n')
=======
import os
import torch
import numpy as np
from Plots import Plots
from laserhockey import hockey_env as h_env
import sys
from DDPGAgent import DDPGAgent
from Parser import opts
#from Evaluater import Evaluater

DIR = os.getcwd()
MODEL_DIR = os.getcwd() + '/weights/'
PRINT = True

# Training
def train(env_name):

    # get information from parser
    NAME = opts.job_id
    RESULT_LOCATION = opts.experiment
    max_episodes = opts.max_episodes
    max_steps = opts.max_steps
    epsilon = opts.epsilon
    min_epsilon = opts.min_epsilon
    epsilon_decay = opts.epsilon_decay
    gamma = opts.gamma
    tau = opts.tau
    milestone = opts.milestone
    eval_interval = opts.eval_interval
    batch_size = opts.batch_size
    update_target_every = opts.update_target_every
    iter_train = opts.iter_train

    # initialize environment
    weak_opponent = True

    game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    if opts.mode == 1:
        game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        game_mode == h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING

    env = h_env.HockeyEnv(mode=game_mode)
    action_dim = env.action_space.shape[0]

    # initalize agent
    agent =  DDPGAgent(env.observation_space, env.action_space, action_dim, tau=tau, gamma=gamma,
                     hidden_size_actor=opts.hidden_size_actor, lr_actor = opts.lr_actor,
                     hidden_size_critic=opts.hidden_size_critic, lr_critic=opts.lr_critic,
                     update_target_every = opts.update_target_every)
    if opts.model is not None:
        agent.load_model(opts.model)

    # initialize return information

    rewards_res = []
    actor_loss_res = []
    critic_loss_res = []
    lengths = []
    total_wins = 0
    total_losses = 0

    for episode in range(1, max_episodes+1):
        player2 = h_env.BasicOpponent(weak=True)
        state, _ = env.reset()

        puk_touched_counter = 1
        total_reward = 0
        wins_interval = 0
        loss_interval = 0

        agent.reset()
        epsilon = max(epsilon_decay * epsilon, min_epsilon)
        done = False

        for step in range(1, max_steps+1):
            obs_player2 = env.obs_agent_two()
            action1 = agent.act(state, eps = epsilon)
            action2 = player2.act(obs_player2)

            next_state, reward, done, trunc, info = env.step(np.hstack([action1, action2]))
            puk_touched_counter += info["reward_touch_puck"]
            reward = 3 * reward + info["reward_closeness_to_puck"] + 3 * (
                        info["reward_puck_direction"] + (5 / puk_touched_counter) * info["reward_touch_puck"])
            total_reward += reward
            agent.store_transition(state, action1, next_state, reward, done)
            state = next_state

            if done or trunc:
                break

        if info['winner'] == 1:
            wins_interval +=1
        elif info['winner'] == -1:
            loss_interval += 1

        actor_loss, critic_loss = agent.train(batch_size)
        actor_loss_res.append(actor_loss)
        critic_loss_res.append(critic_loss)
        rewards_res.append(total_reward)
        lengths.append(step)

        #logging
        if episode % eval_interval == 0:
            avg_reward = np.mean(reward[-eval_interval:])
            avg_length = int(np.mean(lengths[-eval_interval:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
            print(f'Wins: {loss_interval/eval_interval}, Losses: {wins_interval/eval_interval}')
            loss_interval = 0
            eval_interval = 0
            total_wins += wins_interval
            total_losses += loss_interval

        if (episode % milestone == 0):
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{RESULT_LOCATION}/{NAME}_{env_name}_actor_m{opts.mode}_"
                                                 f"h-size{opts.hidden_size_actor}_lr{opts.lr_actor}_e{episode}.pth")
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{RESULT_LOCATION}/{NAME}_{env_name}_critic_m{opts.mode}_"
                                                 f"h-size{opts.hidden_size_critic}_lr{opts.lr_critic}_e{episode}.pth")
    return {"reward": rewards_res, "actor_loss": actor_loss_res, "critic_loss": critic_loss_res,
            "stats": [["wins", "losses", "draws"], [total_wins, total_losses, max_episodes-(total_wins+total_losses)]]}

if __name__ == "__main__":

    train_result = train("Hockey")
    plots_train = Plots(DIR+"/results/train_results")
    plots_train.save_results(train_result)
    plots_train.plot_res(train_result["actor_loss"],  running_mean = int(opts.max_episodes/100), title=f'train_actor_loss')
    plots_train.plot_res(train_result["critic_loss"], running_mean = int(opts.max_episodes/100), title=f'train_critic_loss')
    plots_train.plot_res(train_result["reward"], running_mean = int(opts.max_episodes/100), title=f'train_reward')



    print(f'########## Hyperparameter: ####################'
          f'max_episodes : {opts.max_episodes} \n' 
          f'mode: {opts.mode} \n'
          f'epsilon: {opts.epsilon} \n'
          f'epsilon_decay: {opts.epsilon_decay} \n'
          f'observation noice: {opts.obs_noice} \n'
          f'tau: {opts.tau}\n'
          f'gamma: {opts.gamma}\n'
          f'lr_actor: {opts.lr_actor}\n'
          f'hidden_size_actor: {opts.hidden_size_actor}\n'
          f'lr_critic: {opts.lr_critic}\n'
          f'hidden_size_critic: {opts.hidden_size_critic}\n'
          f'batch_size: {opts.batch_size}\n')
>>>>>>> 2fe89b120e8ed3f5140ed9858fa1143bc77a9ce3
