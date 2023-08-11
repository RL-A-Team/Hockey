import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from SAC.utils import plot_percentages

if __name__ == '__main__':

    f1 = False
    f2 = False
    f3 = False
    f4 = True
    f5 = False
    f6 = False

    if f1:
        # STANDARD REWARD
        different_alphas_dirs = ['eval/standard_reward/01_alpha_autotune',
                                 'eval/standard_reward/02_alpha_02',
                                 'eval/standard_reward/03_alpha_04',
                                 'eval/standard_reward/04_alpha_06',
                                 'eval/standard_reward/05_alpha_08',
                                 'eval/standard_reward/06_autotun_prb']

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        for i, alpha in enumerate(['autotune', 0.2, 0.4, 0.6, 0.8]):
            dir = different_alphas_dirs[i]

            kpi_files = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.csv'))]
            print(dir)
            print(len(kpi_files))
            kpis = pd.read_csv(kpi_files[0])
            for file in kpi_files[1:]:
                kpis = kpis.append(pd.read_csv(file))

            eval_percent_win = kpis['eval_percent_win'].values
            eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
            eval_percent_lose = kpis['eval_percent_lose'].values
            eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

            ax1.plot(eval_percent_win, label=alpha)
            #ax.scatter(eval_percent_win, label=alpha)
            ax2.plot(eval_percent_lose, label=alpha)
            #ax.scatter(eval_percent_lose, label=alpha)

            num_ones = 0
            win_percentages = []
            for i, val in enumerate(kpis['stats_win'].values, 1):
                num_ones += val
                win_percent = (num_ones / i)
                win_percentages.append(win_percent)

            ax3.plot(range(1, len(kpis['stats_win'].values) + 1), win_percentages, label=alpha)

        ax1.set_ylim([0,1])
        ax1.legend()
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.set_title('Percentage of wins')

    if f2:
        different_alphas_dirs = ['eval/standard_reward/07_alpha_autotune_lr1e-4',
                                 'eval/standard_reward/08_alpha_02_lr1e-4',
                                 'eval/standard_reward/09_alpha_04_lr1e-4',
                                 'eval/standard_reward/10_auotun_prb_lr1e-4',
                                 'eval/standard_reward/11_alpha_autotune_lr1e-5',
                                 'eval/standard_reward/12_alpha_02_lre1-5',
                                 'eval/standard_reward/13_alpha_04_lr1e-5',
                                 'eval/standard_reward/14_autotun_prb_lr1e-5',]

        fig, ax = plt.subplots()

        for i, alpha in enumerate(['autotune 1e-4', '0.2 1e-4', '0.4 1r-4', 'PRB 1e-4', 'autotune 1e-5', '0.2 1e-5',
                                   '0.4 1e-5', 'PRB 1e-5']):
            dir = different_alphas_dirs[i]

            kpi_files = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.csv'))]
            print(dir)
            print(len(kpi_files))
            kpis = pd.read_csv(kpi_files[0])
            for file in kpi_files[1:]:
                kpis = kpis.append(pd.read_csv(file))

            eval_percent_win = kpis['eval_percent_win'].values
            eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
            eval_percent_lose = kpis['eval_percent_lose'].values
            eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

            ax.plot(eval_percent_win, label=alpha)
            # ax.scatter(eval_percent_win, label=alpha)
            # ax.plot(eval_percent_lose, label=alpha)
            # ax.scatter(eval_percent_lose, label=alpha)

        ax.set_ylim([0, 1])
        ax.legend()

    if f3:
        different_alphas_dirs = ['eval/test_rewards/r-1/1e-4',
                                 'eval/test_rewards/r-1/1e-5',
                                 #'eval/test_rewards/r1',
                                 #'eval/test_rewards/r2',
                                 #'eval/test_rewards/r3',
                                 #'eval/test_rewards/r4',
                                 #'eval/test_rewards/r5',
                                 #'eval/test_rewards/r6/1e-4',
                                 'eval/test_rewards/r6/1e-5',
                                 #'eval/test_rewards/r7/1e-4',
                                 'eval/test_rewards/r7/1e-5',
                                 'eval/test_rewards/r8',
                                 'eval/test_rewards/r9',
                                    ]

        fig, ax = plt.subplots()

        for i, alpha in enumerate(['r-1 1e-4', 'r-1 1e-5',
                                   #'r1', 'r2', 'r3', 'r4', 'r5',
                                   #'r6 1e-4',
                                   'r6 1e-5',
                                    #'r7 1e-4',
                                   'r7 1e-5',
                                   'r8', 'r9'
                                   ]):
            dir = different_alphas_dirs[i]

            kpi_files = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.csv'))]
            print(dir)
            print(len(kpi_files))
            kpis = pd.read_csv(kpi_files[0])
            for file in kpi_files[1:]:
                kpis = kpis.append(pd.read_csv(file))

            eval_percent_win = kpis['eval_percent_win'].values
            eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
            eval_percent_lose = kpis['eval_percent_lose'].values
            eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

            ax.plot(eval_percent_win, label=alpha)
            # ax.scatter(eval_percent_win, label=alpha)
            # ax.plot(eval_percent_lose, label=alpha)
            # ax.scatter(eval_percent_lose, label=alpha)

        ax.set_ylim([0, 1])
        ax.legend()

    if f4:
        # STANDARD REWARD
        dir = '../eval/r19'

        #fig = plt.figure()
        #gs = fig.add_gridspec(2, 1, hspace=0.15, wspace=0.08)
        #(ax1, ax2) = gs.subplots(sharex='col')
        fig, ax1 = plt.subplots()
        kpi_files = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.csv'))]
        kpi_files.sort()
        kpis = pd.read_csv(kpi_files[0])
        for file in kpi_files[1:]:
            kpis = kpis.append(pd.read_csv(file))

        eval_percent_win = kpis['eval_percent_win'].values
        eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
        eval_percent_lose = kpis['eval_percent_lose'].values
        eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

        ax1.plot(np.arange(0, len(eval_percent_win) * 500, 500), eval_percent_win, color='#141414', )
        #ax2.plot(np.arange(0, len(eval_percent_lose) * 500, 500), eval_percent_lose,)

        x = [0,12000]
        y1 = [0,0]
        y2 = [1,1]
        ax1.fill_between(x, y1, y2, color='C0', alpha=0.15)
        #ax2.fill_between(x, y1, y2, color='C0', alpha=0.15)
        x = [12000, 21000]
        ax1.fill_between(x, y1, y2, color='C1', alpha=0.15)
        #ax2.fill_between(x, y1, y2, color='C1', alpha=0.15)
        x = [21000, 24000]
        ax1.fill_between(x, y1, y2, color='C0', alpha=0.15)
        #ax2.fill_between(x, y1, y2, color='C0', alpha=0.15)
        x = [24000, 25000]
        ax1.fill_between(x, y1, y2, color='C2', alpha=0.15)
        #ax2.fill_between(x, y1, y2, color='C2', alpha=0.15)

        ax1.scatter(np.arange(0, len(eval_percent_win) * 500, 500), eval_percent_win, color="#141414", )
        #ax2.scatter(np.arange(0, len(eval_percent_lose) * 500, 500), eval_percent_lose)

        num_ones = 0
        win_percentages = []
        for i, val in enumerate(kpis['stats_win'].values, 1):
            num_ones += val
            win_percent = (num_ones / i)
            win_percentages.append(win_percent)

        # ax3.plot(range(1, len(kpis['stats_win'].values) + 1), win_percentages, label=alpha)

        ax1.set_ylim([0, 1])
        #ax1.legend()
        ax1.set_xlabel('Training episodes')
        # ax1.set_xticklabels([])
        ax1.set_ylabel('Probability')
        ax1.set_axisbelow(True)
        ax1.set_title('Winning probability')
        ax1.grid()
        #ax2.set_ylim([0, 1])
        # ax2.legend()
        #ax2.set_axisbelow(True)
        #ax2.set_xlabel('Training episodes')
        #ax2.set_title('b) Losing probability')
        #ax2.grid()
        #ax2.legend(loc='upper right')
        #for ax in fig.get_axes():
        #    ax.label_outer()

    if f5:
        # STANDARD REWARD
        files = ['experiments/losses/l1.csv',
                 'experiments/losses/l2.csv']

        #fig, (ax1, ax2) = plt.subplots(1, 2, )
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.08)
        (ax1, ax2) = gs.subplots(sharex='col')
        #fig3, ax3 = plt.subplots()

        for i, alpha in enumerate(['L1', 'L2']):
            kpis = pd.read_csv(files[i])

            eval_percent_win = kpis['eval_percent_win'].values
            eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
            eval_percent_lose = kpis['eval_percent_lose'].values
            eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

            ax1.plot(np.arange(0,len(eval_percent_win)*500,500), eval_percent_win)
            ax2.plot(np.arange(0,len(eval_percent_lose)*500,500), eval_percent_lose, label=alpha)
            ax1.scatter(np.arange(0, len(eval_percent_win) * 500, 500), eval_percent_win, cmap="Dark2",)
            ax2.scatter(np.arange(0,len(eval_percent_lose)*500,500), eval_percent_lose)

            num_ones = 0
            win_percentages = []
            for i, val in enumerate(kpis['stats_win'].values, 1):
                num_ones += val
                win_percent = (num_ones / i)
                win_percentages.append(win_percent)

            #ax3.plot(range(1, len(kpis['stats_win'].values) + 1), win_percentages, label=alpha)

        ax1.set_ylim([0,1])
        #ax1.legend()
        ax1.set_xlabel('Training episodes')
        #ax1.set_xticklabels([])
        ax1.set_ylabel('Probability')
        ax1.set_axisbelow(True)
        ax1.set_title('a) Winning probability')
        ax1.grid()
        ax2.set_ylim([0, 1])
        #ax2.legend()
        ax2.set_axisbelow(True)
        ax2.set_xlabel('Training episodes')
        ax2.set_title('b) Losing probability')
        ax2.grid()
        ax2.legend(loc='upper right')
        for ax in fig.get_axes():
            ax.label_outer()
        #ax3.set_ylim([0, 1])
        #ax3.legend()
        #ax3.set_title('Percentage of wins')

    if f6:
        # STANDARD REWARD
        files = ['experiments/alphas/0-1.csv',
                 'experiments/alphas/0-2.csv',
                 'experiments/alphas/0-3.csv',
                 'experiments/alphas/0-4.csv',
                 'experiments/alphas/0-5.csv',
                 'experiments/alphas/autotune.csv',
                 ]

        #fig, (ax1, ax2) = plt.subplots(1, 2, )
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0.08)
        (ax1, ax2) = gs.subplots(sharex='col')
        #fig3, ax3 = plt.subplots()

        for i, alpha in enumerate([r'$\alpha=0.1$', r'$\alpha=0.1$', r'$\alpha=0.3$', r'$\alpha=0.4$', r'$\alpha=0.5$', 'autotune']):
            kpis = pd.read_csv(files[i])

            eval_percent_win = kpis['eval_percent_win'].values
            eval_percent_win = eval_percent_win[~np.isnan(eval_percent_win)]
            eval_percent_lose = kpis['eval_percent_lose'].values
            eval_percent_lose = eval_percent_lose[~np.isnan(eval_percent_lose)]

            ax1.plot(np.arange(0,len(eval_percent_win)*500,500), eval_percent_win)
            ax2.plot(np.arange(0,len(eval_percent_lose)*500,500), eval_percent_lose, label=alpha)
            ax1.scatter(np.arange(0, len(eval_percent_win) * 500, 500), eval_percent_win, cmap="Dark2",)
            ax2.scatter(np.arange(0,len(eval_percent_lose)*500,500), eval_percent_lose)

            num_ones = 0
            win_percentages = []
            for i, val in enumerate(kpis['stats_win'].values, 1):
                num_ones += val
                win_percent = (num_ones / i)
                win_percentages.append(win_percent)

            #ax3.plot(range(1, len(kpis['stats_win'].values) + 1), win_percentages, label=alpha)

        ax1.set_ylim([0,1])
        #ax1.legend()
        ax1.set_xlabel('Training episodes')
        #ax1.set_xticklabels([])
        ax1.set_ylabel('Probability')
        ax1.set_axisbelow(True)
        ax1.set_title('a) Winning probability')
        ax1.grid()
        ax2.set_ylim([0, 1])
        #ax2.legend()
        ax2.set_axisbelow(True)
        ax2.set_xlabel('Training episodes')
        ax2.set_title('b) Losing probability')
        ax2.grid()
        ax2.legend(loc='upper right')
        for ax in fig.get_axes():
            ax.label_outer()
        #ax3.set_ylim([0, 1])
        #ax3.legend()
        #ax3.set_title('Percentage of wins')

    plt.tight_layout()
    plt.show()

