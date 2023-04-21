# coding=UTF-8
from environment_model import Environment
from command_parser import command_parse
from agent_model import State
from persistent_model import Persistent
from logger import Logger
# from analysis import Analysis
from data.data_clean import DataCleaner
import tensorflow as tf
import sys


class Main:
    def __init__(self, kwargs):
        tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
        if 'cleaner' in kwargs:
            cleaner = kwargs['cleaner']
        else:
            cleaner = DataCleaner()
        parameters = command_parse(sys.argv[1:], cleaner, kwargs)
        # self.analysis = parameters['analysis']
        Persistent.init(train=parameters['train'],
                        continue_train=parameters['continue_train'],
                        compare=parameters['compare'],
                        compare_method=parameters['compare_method'],
                        task_assignment_method=parameters['task_assignment_policy'],
                        no_uav=parameters['no_uav'],
                        directory=parameters['prefix'],
                        suffix=parameters['suffix'])
        State.init(sensor_number=parameters['sensor_number'],
                   cell_size=cleaner.cell_limit,
                   max_energy=parameters['uav_energy'])
        Logger.init(console_log=parameters['console_log'],
                    file_log=parameters['file_log'],
                    directory=parameters['prefix'])
        self.env = Environment(train=parameters['train'],
                               continue_train=parameters['continue_train'],
                               compare=parameters['compare'],
                               compare_method=parameters['compare_method'],
                               sensor_number=parameters['sensor_number'],
                               worker_number=parameters['worker_number'],
                               max_episode=parameters['max_episode'],
                               batch_size=parameters['batch_size'],
                               epsilon_decay=parameters['epsilon_decay'],
                               learn_rate=parameters['learn_rate'],
                               gamma=parameters['gamma'],
                               detail=parameters['detail'],
                               seed=parameters['seed'],
                               mali_rate=parameters['malicious'],
                               win_len=parameters['windows_length'],
                               pho=parameters['pho'],
                               task_assignment_policy=parameters['task_assignment_policy'],
                               no_uav=parameters['no_uav'],
                               # assignment_reduce_rate=parameters['reduce_rate'],
                               cost_limit=parameters['cost_limit'],
                               max_energy=parameters['uav_energy'],
                               basic_reward_for_worker=parameters['basic_reward_for_worker'],
                               max_bid_for_worker=parameters['max_bid_for_worker'],
                               cleaner=cleaner)

    def start(self):
        self.env.start()

    def end(self):
        Persistent.close()


if __name__ == '__main__':
    # CCPP Greedy RR
    # cleaner = DataCleaner(
    #              x_limit=6,
    #              y_limit=6,
    #              x_range=[0, 2000],  # 可以是角度，也可以是距离
    #              y_range=[0, 1800],
    #              range_is_angle=False, # 很重要的参数，确定是角度的话需要乘以地球系数
    #              Norm=True,
    #              Norm_centers=[[500, 600], [1400, 1100]],
    #              Norm_centers_ratio=[0.4, 0.6], # 每一个分布中心所占比率
    #              Norm_sigma=[1, 1],       # 正态分布的方差
    #              Norm_gain=[400, 600],        # 正态分布系数，用于控制器辐射半径大小
    #              No_data_set_need=True)
    cleaner = DataCleaner(
                 # x_limit=6,
                 # y_limit=6,
                 # x_range=[0, 2000],  # 可以是角度，也可以是距离
                 # y_range=[0, 1800],
                 # range_is_angle=False, # 很重要的参数，确定是角度的话需要乘以地球系数
                 # Norm=False,
                 # Norm_centers=[],
                 # Norm_centers_ratio=[0.4, 0.6], # 每一个分布中心所占比率
                 # Norm_sigma=[1, 1],       # 正态分布的方差
                 # Norm_gain=[400, 600],        # 正态分布系数，用于控制器辐射半径大小
                 # No_data_set_need=True
    )
    data = {'sensor_number': 5000,
            'worker_number': 1000,
            'console_log': True,
            'gamma': 0.9,
            'learn_rate': 0.001,
            'uav_energy': 77,
            'max_episode': 600,
            'cost_limit': 250,
            # 'task_assignment_policy': 'random',
            'train': True,
            # 'compare': True,
            # 'no_uav': True,
            # 'compare_method': 'RR',
            'cleaner': cleaner}
    # data = {'sensor_number': 5000, 'worker_number': 1000, 'console_log': True,
    #                   'gamma': 0.95, 'learn_rate': 0.001, 'train': False, 'uav_energy': 77,
    #                   'compare':True,
    #                   'compare_method': 'RR',
    #                   'task_assignment_policy': 'genetic',
    #                   'max_episode': 1,
    #                   'cleaner': cleaner}
    processor = Main(data)
    processor.start()
    processor.end()

