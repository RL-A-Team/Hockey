<<<<<<< HEAD
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

    def save_results_in_one(self, result_dict):
        res = []
        temp = []
        with open(f'{self.path}/{self.timestamp}.txt', 'w') as f:
            print(result_dict, file=f)

    def plot_res(self, losses, running_mean, title):
        plt.figure()
        plt.plot(self.running_mean(losses, running_mean), label = title)
        plt.legend()
        plt.savefig(f'{self.path}\\{self.timestamp}_{title}.png')

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
=======
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
>>>>>>> 2fe89b120e8ed3f5140ed9858fa1143bc77a9ce3
        return (cumsum[N:] - cumsum[:-N]) / float(N)