import numpy as np
import pandas as pd


class Analysis:

    def __init__(self, train: bool):
        self.__train = train



    def start(self):
        if self.__train:
            plot_average_aoi_curve(real=True, observation=True)
            plot_average_energy_curve()
            plot_reward_energy_curve()
            plot_train_curve_in_different_parameter()
        else:
            plot_real_aoi_curve(real=True, observation=True)
            plot_cell_aoi_hotmap()




