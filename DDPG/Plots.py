import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Plots():
    def __init__(self, path):
        self.path = path + "/results/"
        self.timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    def save_results(self, result_dict):
        for res in result_dict:
            np.savetxt(f'{self.path}/{self.timestamp}_{res}.txt', result_dict[res], fmt="%s", delimiter='\t')

    def plot_losses(self, losses, title):
        plt.figure()
        plt.plot(range(len(losses)), losses, label = title)
        plt.legend()
        plt.savefig(f'{self.path}\\{self.timestamp}_{title}.png')

    def plot_reward(self, rew, title):
        plt.figure()
        plt.plot(range(len(rew)), rew, label=title)
        plt.legend()
        plt.savefig(f'{self.path}\\{self.timestamp}_{title}.png')
