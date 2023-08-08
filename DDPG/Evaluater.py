from DDPGAgent import DDPGAgent
from laserhockey import hockey_env as h_env
import os
import numpy as np
import time
from Plots import Plots
from Parser import opts
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

DIR = os.getcwd()

class Evaluater():
    def __init__(self, path=None, visualize = True):
        if path == None:
            self.path = os.getcwd() + '/weights/'
        else:
            self.path = path
        self.visualize = visualize

    def evaluate_one_model(self, max_episodes, step_size, model, mode, opponment = None):
        game_mode = h_env.HockeyEnv_BasicOpponent.NORMAL
        if mode == 1:
            game_mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
        else:
            game_mode == h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING

        env = h_env.HockeyEnv(mode=game_mode)
        action_dim = env.action_space.shape[0]

        rewards = []
        won = []
        lost = []
        draws=[]
        total_wins = 0
        total_losses = 0

        # initalize agent
        agent = DDPGAgent(observation_space = env.observation_space, action_space= env.action_space, action_dim=action_dim,
                          tau=opts.tau, gamma=opts.gamma, hidden_size_actor=opts.hidden_size_actor, lr_actor=opts.lr_actor,
                        hidden_size_critic=opts.hidden_size_critic, lr_critic=opts.lr_critic,
                          update_target_every=opts.update_target_every)
        if opts.model is not None:
            agent.load_model(self.path + opts.model)

        for episode in range(1, max_episodes+1):
            # initialize player 2
            total_reward = 0
            if opponment is None:
                player2 = h_env.BasicOpponent(weak=True)
            else:
                player2 = opponment

            state, _ = env.reset()
            for steps in range(1, step_size+1):
                if self.visualize:
                    time.sleep(0.01)
                    env.render("human")

                action1 = agent.act(state, eps=0)
                obs_player2 = env.obs_agent_two()
                action2 = player2.act(obs_player2)
                next_state, reward, done, trunc, info = env.step(np.hstack([action1, action2]))
                total_reward += reward
                state = next_state

                if done or trunc:
                    break

            if info['winner'] == 1:
                total_wins += 1
                won.append(1)
                lost.append(0)
                draws.append(0)

            elif info['winner'] == -1:
                total_losses += 1
                won.append(0)
                lost.append(1)
                draws.append(0)

            else:
                won.append(0)
                lost.append(0)
                draws.append(1)

            rewards.append(total_reward)
            print(rewards)
        return {"eval_reward": rewards, "won": won, "lost": lost, "draws": draws}


    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)


    def evaluate_per_directory(self, max_episodes, step_size, model_type, group_by = None, groub_by_name = None, opponment = None):
        models = os.listdir(self.path)
        hidden_sizes = []
        modes = []
        num = 0
        res = []
        result_dict = {"group": [], "mean_reward": [], "var_reward":[],
        "mean_wins": [], "var_wins":[], "mean_loss": [], "var_loss": []}
        plt.rcParams.update({'font.size': 10})
        fig,axs = plt.subplots(2, 2, figsize=(12, 11))

        for group in group_by:
            # some regex to evaluate the right models and put them in groups
            total_wins = 0
            total_losses = 0
            won = np.zeros([max_episodes, 2])
            lost=np.zeros([max_episodes, 2])
            draws=np.zeros([max_episodes, 2])
            rewards = np.zeros([max_episodes, 2])
            model_num = 0
            for model in models:
                # actor vs. actorParameterized vs. critic
                if (model.find(model_type) == -1):
                    continue

                # groub by variabele to test
                if ((group_by is not None) and (model.find("test"+str(group)) == -1)):
                    continue
                start = model.find("h-size")+6
                end = model[start:].find("_") + start
                #hidden_sizes.append(int(model[start:end]))
                hidden_sizes = opts.hidden_size_actor
                modes.append(int(model[model.find("_m")+2]))

                results = self.evaluate_one_model(max_episodes, step_size, model, mode=modes[num])
                length = len([results["eval_reward"]])
                rewards[:max_episodes, model_num] = results["eval_reward"][-max_episodes:]
                draws[:max_episodes, model_num] = results["draws"][-max_episodes:]
                lost[:max_episodes, model_num] = results["lost"][-max_episodes:]
                won[:max_episodes, model_num] = results["won"][-max_episodes:]
                model_num += 1

            axs[0, 0].plot(self.running_mean(np.mean(rewards, axis=1), 10), label = str(group))
            legend1 = axs[0, 0].legend(loc='upper left', fontsize=9)
            axs[0, 1].plot(self.running_mean(np.sum(won, axis=1), 10), label = str(group))
            legend2 = axs[0, 1].legend(loc='upper left', fontsize=9)
            axs[1, 0].plot(self.running_mean(np.sum(lost, axis=1), 10), label = str(group))
            legend3 = axs[1, 0].legend(loc='upper left', fontsize=9)
            axs[1, 1].plot(self.running_mean(np.sum(draws, axis=1), 10), label = str(group))
            legend4 = axs[1, 1].legend(loc='upper left', fontsize=9)
            num += 1

            result_dict["group"].append(group)
            result_dict["mean_reward"].append(np.mean(rewards))
            result_dict["var_reward"].append(np.var(rewards))
            result_dict["mean_wins"].append(np.mean(won))
            result_dict["var_wins"].append(np.var(won))
            result_dict["mean_loss"].append(np.mean(lost))
            result_dict["var_loss"].append(np.var(lost))


        # grouping
        plots_train = Plots(DIR + "/results/train_eval/tau/"+str(opts.experiment))
        #print(result_dict)
        plots_train.save_results_in_one(result_dict)

        extent = self.full_extent(axs[0, 0]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{DIR+"/results/train_eval/tau/rewards_"+str(opts.experiment)}_{group_by}.png', bbox_inches=extent)
        extent = self.full_extent(axs[0, 1]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{DIR+"/results/train_eval/tau/wins_"+str(opts.experiment)}_{group_by}.png', bbox_inches=extent)
        extent = self.full_extent(axs[1, 0]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{DIR+"/results/train_eval/tau/lossses_"+str(opts.experiment)}_{group_by}.png', bbox_inches=extent)
        extent = self.full_extent(axs[1, 1]).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{DIR+"/results/train_eval/tau/draws_"+str(opts.experiment)}_{group_by}.png', bbox_inches=extent)

        fig.savefig(f'{DIR+"/results/train_eval/tau/"+str(opts.experiment)}_{group_by}.png', bbox_inches=extent)

    def full_extent(self, ax, pad=0.0):
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])

        return bbox.expanded(1.0 + pad, 1.0 + pad)


if __name__ == "__main__":
    evaluater = Evaluater(os.getcwd()+'/weights/tau/', visualize=False)

    evaluater.evaluate_per_directory(max_episodes=100, step_size=250,
                                     model_type="actor", group_by=[0.0, 0.5, 1.0],
                                     groub_by_name="tau")