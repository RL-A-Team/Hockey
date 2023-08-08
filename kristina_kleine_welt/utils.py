import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from sac import SACAgent


def cal_running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def longest(lists):
    """ get the length of the longst list

    :param lists: list[list[Object]]
    :return int
    """
    max = 0
    for list in lists:
        if len(list) > max:
            max = len(list)

    return max


def fill_nan(lists):
    """fill shorter lists with nans such that all lists have the same length (as the longest lists)

    :param lists: list[list[Object]]
    :return tuple(lists[0], ..., lists[n])
    """
    desired_length = longest(lists)

    for i, list in enumerate(lists):
        diff = desired_length - len(list)
        if diff > 0:
            add = [None] * diff
            list.extend(add)

    return tuple(lists)


def evaluation_plot(values, title, running_mean):
    """create plot of the running mean values"""
    if running_mean:
        values = cal_running_mean(np.asarray(values), 500)

    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)


def plot_actor_critic_losses(critic1_losses, critic2_losses,
                             actor_losses, alpha_losses, running_mean=True):
    """create plot of each of the losses"""
    evaluation_plot(critic1_losses, "Critic 1 loss", running_mean)
    evaluation_plot(critic2_losses, "Critic 2 loss", running_mean)
    evaluation_plot(actor_losses, "Actor loss", running_mean)
    evaluation_plot(alpha_losses, "Alpha loss", running_mean)


def plot_wins_loses(stats_win, stats_lose):
    """plot cumsum of wins and loses"""
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(stats_win), color='green', label='Win')
    ax.plot(np.cumsum(stats_lose), color='red', label='Lose')
    ax.set_title("Win vs. lose")
    ax.legend()


def save_statistics(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose, mean_rewards,
                    mean_win, mean_lose, eval_percent_win, eval_percent_lose, filename):
    """ Save statistics to :param: filename    """

    # bring all statistics to the same length
    critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose, mean_rewards, mean_win, \
    mean_lose, eval_percent_win, eval_percent_lose = \
        fill_nan([critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose, mean_rewards,
                  mean_win, mean_lose, eval_percent_win, eval_percent_lose])

    stats = pd.DataFrame({'critic1_losses': critic1_losses,
                          'critic2_losses': critic2_losses,
                          'actor_losses': actor_losses,
                          'alpha_losses': alpha_losses,
                          'stats_win': stats_win,
                          'stats_lose': stats_lose,
                          'mean_rewards': mean_rewards,
                          'mean_win': mean_win,
                          'mean_lose': mean_lose,
                          'eval_percent_win': eval_percent_win,
                          'eval_percent_lose': eval_percent_lose})
    stats.to_csv(f'{filename}.csv', index=False)
    print(f"Statistics saved in file {filename}.csv")


def plot_percentages(stats_win, stats_lose):
    """ Plot win/lose percentage over time

    :param stats_win: list[float]
        1 if our agent won, 0 else
    :param stats_lose: list[float]
        1 if our agent lost, 0 else
    """
    num_ones = 0
    win_percentages = []

    for i, val in enumerate(stats_win, 1):
        num_ones += val
        win_percent = (num_ones / i)
        win_percentages.append(win_percent)

    num_ones = 0
    lose_percentages = []
    for i, val in enumerate(stats_lose, 1):
        num_ones += val
        lose_percent = (num_ones / i)
        lose_percentages.append(lose_percent)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(stats_win) + 1), win_percentages, color='green')
    ax.plot(range(1, len(stats_lose) + 1), lose_percentages, color='red')
    ax.set_title('Percentage of wins and losses')


def plot_eval(eval_percent_win, eval_percent_lose):
    """ Plot the evaluation results

    :param eval_percent_win: list[float]
        mean number of win in evaluation iteration
    :param eval_percent_lose: list[float]
        mean number of lose in evaluation iteration
    """
    fig, ax = plt.subplots()
    ax.plot(eval_percent_win, color='green')
    ax.plot(eval_percent_lose, color='red')
    ax.set_ylim([0, 1])
    ax.set_title('Evaluation results')


def save_multi_image(filename):
    """ Saves all produced figures as pdf in the given filename

        :param filename: str
            Filename where the plots should be stored

        Source:
        https://www.tutorialspoint.com/saving-multiple-figures-to-one-pdf-file-in-matplotlib
    """

    pp = PdfPages(filename + '.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
        plt.close(fig)
    pp.close()
    print(f"Plots saved in file {filename}.pdf'")


def save_evaluation_results(critic1_losses, critic2_losses,
                            actor_losses, alpha_losses, stats_win, stats_lose, mean_rewards, mean_win, mean_lose,
                            eval_percent_win, eval_percent_lose,
                            model: SACAgent):
    """ Save and plot results.

    :param critic1_losses: list[float]
        loss values of the critic1 network
    :param critic2_losses: list[float]
        loss values of the critic2 network
    :param actor_losses: list[float]
        loss values of the actor network
    :param alpha_losses: list[float]
        loss values of the alpha network
    :param stats_win: list[float]
        1 if our agent won, 0 else
    :param stats_lose: list[float]
        1 if our agent lost, 0 else
    :param mean_rewards: list[float]
        mean reward in episode
    :param mean_win: list[float]
        mean number of win in episode
    :param mean_lose: list[float]
        mean number of lose in episode
    :param eval_percent_win: list[float]
        mean number of win in evaluation iteration
    :param eval_percent_lose: list[float]
        mean number of lose in evaluation iteration
    :param model: sac.SACAgent
        trained model
    :return:
    """
    # plot losses
    plot_actor_critic_losses(critic1_losses, critic2_losses, actor_losses, alpha_losses, False)

    # plot rewards
    evaluation_plot(mean_rewards, "Mean reward per episode", True)

    # plot mean wins and loses
    evaluation_plot(mean_win, "Mean wins per episode", True)
    evaluation_plot(mean_lose, "Mean lose per episode", True)
    plot_wins_loses(stats_win, stats_lose)
    plot_percentages(stats_win, stats_lose)

    # plot evaluation results
    plot_eval(eval_percent_win, eval_percent_lose)

    # unique identifier for filenames (timestamp and random integer)
    dt_now = f'{datetime.now().strftime("%Y%m%dT%H%M%S")}_{np.random.randint(0, 100000)}'

    # save statistics
    s_filename = f'eval/sac_stats_{dt_now}'
    save_statistics(critic1_losses, critic2_losses, actor_losses, alpha_losses, stats_win, stats_lose, mean_rewards,
                    mean_win, mean_lose, eval_percent_win, eval_percent_lose, s_filename)

    # save plots
    p_filename = f'eval/sac_plots_{dt_now}'
    save_multi_image(p_filename)

    # save model
    m_filename = f'models/sac_model_{dt_now}.pkl'
    pickle.dump(model, open(m_filename, 'wb'))

    print('')
    print('--------------------------------------')
    print('COPY COMMANDS')
    print(f'scp stud54@tcml-master01.uni-tuebingen.de:~/Hockey/{s_filename}.csv .')
    print(f'scp stud54@tcml-master01.uni-tuebingen.de:~/Hockey/{p_filename}.pdf .')
    print(f'scp stud54@tcml-master01.uni-tuebingen.de:~/Hockey/{m_filename} .')
    print('--------------------------------------')
