import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # STANDARD REWARD
    different_alphas_dirs = ['SAC/eval/standard_reward/01_alpha_autotune',
                             'SAC/eval/standard_reward/02_alpha_02',
                             'SAC/eval/standard_reward/03_alpha_04',
                             'SAC/eval/standard_reward/04_alpha_06',
                             'SAC/eval/standard_reward/05_alpha_08']
    
    for i, alpha in enumerate(['autotune', 0.2, 0.4, 0.6, 0.8]):
