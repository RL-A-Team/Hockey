import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

if __name__ == '__main__':
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

    plt.show()

