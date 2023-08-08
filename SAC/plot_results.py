import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

if __name__ == '__main__':

    f1 = False
    f2 = False
    f3 = True

    if f1:
        # STANDARD REWARD
        different_alphas_dirs = ['eval/standard_reward/01_alpha_autotune',
                                 'eval/standard_reward/02_alpha_02',
                                 'eval/standard_reward/03_alpha_04',
                                 'eval/standard_reward/04_alpha_06',
                                 'eval/standard_reward/05_alpha_08',
                                 'eval/standard_reward/06_autotun_prb']

        fig, ax = plt.subplots()

        for i, alpha in enumerate(['autotune', 0.2, 0.4, 0.6, 0.8, 'PRB']):
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
            #ax.scatter(eval_percent_win, label=alpha)
            #ax.plot(eval_percent_lose, label=alpha)
            #ax.scatter(eval_percent_lose, label=alpha)

        ax.set_ylim([0,1])
        ax.legend()

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
                                 'eval/test_rewards/r6/1e-4',
                                 'eval/test_rewards/r6/1e-5',
                                 'eval/test_rewards/r7/1e-4',
                                 'eval/test_rewards/r7/1e-5',]
                                 #'eval/test_rewards/r8',
                                 #'eval/test_rewards/r9',]

        fig, ax = plt.subplots()

        for i, alpha in enumerate(['r-1 1e-4', 'r-1 1e-5',
                                   #'r1', 'r2', 'r3', 'r4', 'r5',
                                   'r6 1e-4', 'r6 1e-5',
                                   'r7 1e-4', 'r7 1e-5',
                                   #'r8', 'r9'
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

    plt.show()

