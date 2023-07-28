import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from sac import SACAgent


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def evaluation_plot(values, title, running_mean):
    """create plot of the running mean values"""
    if running_mean:
        values = running_mean(np.asarray(values), 500)

    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)


def plot_actor_critic_losses(critic1_losses, critic2_losses,
                             actor_losses, running_mean = True):
    """create plot of each of the losses"""
    evaluation_plot(critic1_losses, "Critic 1 loss", running_mean)
    evaluation_plot(critic2_losses, "Critic 2 loss", running_mean)
    evaluation_plot(actor_losses, "Actor loss", running_mean)


def plot_wins_loses(stats_win, stats_lose):
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(stats_win), color='green', label='Win')
    ax.plot(np.cumsum(stats_lose), color='red', label='Lose')
    ax.set_title("Win vs. lose")
    ax.legend()


def save_statistics(critic1_losses, critic2_losses,
                    actor_losses, stats_win, stats_lose,
                    filename):
    stats = pd.DataFrame({'critic1_losses': critic1_losses,
                          'critic2_losses': critic2_losses,
                          'actor_losses': actor_losses,
                          'stats_win': stats_win,
                          'stats_lose': stats_lose})
    stats.to_csv(f'{filename}.csv', index=False)
    print(f"Statistics saved in file {filename}.csv")


def save_evaluation_results(critic1_losses, critic2_losses,
                            actor_losses, stats_win, stats_lose,
                            model: SACAgent, running_mean=True):
    plot_actor_critic_losses(critic1_losses, critic2_losses, actor_losses, running_mean)
    plot_wins_loses(stats_win, stats_lose)

    dt_now = datetime.now().strftime("%Y%m%dT%H%M%S")
    save_statistics(critic1_losses, critic2_losses, actor_losses, stats_win, stats_lose, f'eval/sac_stats_{dt_now}')
    save_multi_image(f'eval/sac_plots_{dt_now}')

    # save model
    pickle.dump(model, open(f'models/sac_model_{dt_now}.pkl', 'wb'))


def save_multi_image(filename):
    """ Saves all produced figures as pdf in the given filename

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