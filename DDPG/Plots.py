import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Plots():
    def __init__(self, path):
        self.path = path
        self.timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    def save_results(self, result_dict):
        for res in result_dict:
            np.savetxt(f'{self.path}/{self.timestamp}_{res}.txt', result_dict[res], fmt="%s", delimiter='\t')

    def plot_res(self, losses, running_mean, title):
        plt.figure()
        plt.plot(self.running_mean(losses, running_mean), label = title)
        plt.legend()
        plt.savefig(f'{self.path}\\{self.timestamp}_{title}.png')

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)