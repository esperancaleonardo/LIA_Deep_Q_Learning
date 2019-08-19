import matplotlib.pyplot as plt
import math, numpy as np
import random
import os

base_dir = os.getcwd()



class Results(object):
    """docstring for Results."""

    def __init__(self):
        self.rewards_list = []
        self.reward_fig = 0
        self.epsilon_list = []
        self.epsilon_fig = 1
        self.steps_list = []
        self.steps_fig = 2
        self.done_counter_list = []
        self.done_fig = 3
        self.mse_values = []
        self.mse_fig = 4
        self.lost_steps_list = []
        self.lost_fig = 5

    def media_calc(self, data, media):
      data_aux = []
      sum = 0
      for i in range(len(data)):
        if i > 0 and i%media==0:
          data_aux.append(sum/media)
          sum = 0
        else:
          sum += data[i]


      return data_aux

    def plot_media_n(self, data, figure, dir, media, fname, title, ylabel, normalize=False):

      data_aux = self.media_calc(data, media)

      if normalize:
          data = self.normalize_data(data)

      plt.figure(figure)
      plt.autoscale(enable=True, axis='x', tight=True)
      plt.title(title)
      plt.xlabel('Episodes')
      plt.ylabel(ylabel)
      plt.plot(data_aux, 'r')
      plt.savefig(os.path.join(base_dir, "logs", dir, fname+".png"))


    def plot_raw(self, data, figure, dir, fname, title, ylabel, normalize=False):

      if normalize:
          data = self.normalize_data(data)

      plt.figure(figure)
      plt.autoscale(enable=True, axis='x', tight=True)
      plt.title(title)
      plt.xlabel('Episodes')
      plt.ylabel(ylabel)
      plt.plot(data, 'r')
      plt.savefig(os.path.join(base_dir, "logs", dir, fname+".png"))


    def normalize_data(self, data):
      norm = [i/np.sum(data) for i in data]
      return norm
