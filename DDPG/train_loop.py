import os
from Plots import Plots
from laserhockey import hockey_env as h_env
import sys
from DDPG_agent import DDPG_agent

if sys[1] is not None:
    NAME = sys[1]
else:
    NAME = "test"

DIR = os.getcwd()
MODEL_DIR = os.getcwd() + '/weights/'
RENDER = False

# Training
def train(env_name, mode = 0, batch_size=128, max_episodes=10000, max_steps=200, eval_interval = 50,
          epsilon = 0.5, epsilon_decay=0.997, min_epsilon=0.1,
          milestone = 200, tau = 0.005, gamma = 0.99, hidden_size_actor = 64, hidden_size_critic = 64, bufffer_size = 1e4,
          obs_noice = 0.1, model = None):
    env = h_env.HockeyEnv(mode=mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim, hidden_size_actor, hidden_size_critic, buffer_size)
    if model is not None:
        agent.load_mode(model)
    losses = np.zeros((max_episodes, 3))
    episode_reward = 0
    action_space = env.action_space

    for episode in range(max_episodes):

        epsilon = max(epsilon_decay * epsilon, min_epsilon)
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

        for step in range(max_steps):
            if ((episode + 1) % 10 == 0 and RENDER):
                env.render("human")

            obs_player2 = env.obs_agent_two()
            action1 = agent.get_action(state, epsilon = epsilon,
                                       obs_noice = obs_noice, evaluate= False, action_space=action_space)

            if (mode == 1):  # TRAIN_SHOOTING
                action2 = [0, 0, 0, 0]
            else:
                action2 = player2.act(obs_player2)
            next_state, reward, done, _, info = env.step(np.hstack([action1, action2]))
            episode_reward += reward
            reward_per_episode += reward
            puk_touched_counter += info["reward_touch_puck"]

            reward = 3 * reward + info["reward_closeness_to_puck"] + 3 * (
                        info["reward_puck_direction"] + 5 / puk_touched_counter * info["reward_touch_puck"])

            agent.store_experience(state, action1, next_state, reward, done)
            temp_loss = agent.train(batch_size, gamma, tau)
            state = next_state

            if temp_loss != []:
                episode_loss_actor += temp_loss[0]
                episode_loss_critic += temp_loss[1]

            if done:
                break
        if ((episode+1) % eval_interval == 0):
            print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward/eval_interval:.2f}")
            episode_reward = 0

        if (step != 0):
            losses[episode] = [episode_loss_actor / step, episode_loss_critic / step, reward_per_episode]
        else:
            losses[episode] = [0 , 0, episode_reward]

        if ((episode+1) % milestone == 0):
            torch.save(agent.actor.state_dict(), f"{MODEL_DIR}{env_name}-actor-m{mode}-h_size{hidden_size_actor}-e{episode + 1}.pth")
    return losses


if __name__ == "__main__":
    batch_size = 64
    max_episodes = 1000
    max_steps = 200
    eval_interval = 100
    epsilon = 0.4
    epsilon_decay = 0.997
    min_epsilon = 0.1
    tau = 0.005
    gamma = 0.99
    milestone = 200
    hidden_size_actor = 128
    hidden_size_critic = 128
    buffer_size = 100000
    obs_noice = 0.1
    mode = 0 # 0: normal, 1: shooting, 2:defending
    #model = f"{MODEL_DIR}{NAME}Hockey-actor-m{mode}-h_size{hidden_size_actor}-e{max_episodes}.pth"
    model = f"{MODEL_DIR}Hockey-actor-m2-h_size128-e500.pth"

    losses = train("Hockey", mode, batch_size, max_episodes, max_steps, eval_interval, epsilon, epsilon_decay, min_epsilon,
                   milestone, tau, gamma, hidden_size_actor, hidden_size_critic, buffer_size, obs_noice, model=model)
    #mode = 0
    #epsilon = 0.2
    #max_episodes = 1000

    #losses2 = train("Hockey", mode, batch_size, max_episodes, max_steps, eval_interval, epsilon, epsilon_decay,
    #               min_epsilon,
    #               milestone, tau, gamma, hidden_size_actor, hidden_size_critic, buffer_size, obs_noice, model)
    #losses = np.vstack([losses1, losses2])
    plots = Plots(DIR)
    plots.save_results(losses)
    plots.plot_losses(losses)
    plots.plot_reward(losses[:, 2])
