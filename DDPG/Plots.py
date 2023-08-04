import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Plots():
    def __init__(self, path):
        self.path = path + "/results/"

    def save_results(self, losses):
        np.savetxt(f'{self.path}/losses-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.txt', losses)

    def plot_losses(self, losses):
        plt.figure()
        plt.plot(range(len(losses[:,0])), losses[:,0], label = "actor")
        plt.legend()
        plt.savefig(f'{self.path}\\losses_actor-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png')

        plt.figure()
        plt.plot(range(len(losses[:, 0])), losses[:, 1], label="critic")
        plt.legend()
        plt.savefig(f'{self.path}\\losses_critic-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png')

    def plot_reward(self, rew):
        plt.figure()
        plt.plot(range(len(rew)), rew, label="reward")
        plt.legend()
        plt.savefig(f'{self.path}\\reward-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png')
