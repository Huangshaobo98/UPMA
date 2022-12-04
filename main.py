# coding=UTF-8
from environment_model import Environment
from command_parser import command_parse
from agent_model import State
from persistent_model import Persistent
from logger import Logger
from analysis import Analysis
from data.data_clean import DataCleaner
import tensorflow as tf
import sys

if __name__ == '__main__':

    tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
    commands = sys.argv[1:]
    parameters = command_parse()
    cleaner = DataCleaner()
    Persistent.init(analysis=parameters['analysis'],
                    train=parameters['train'],
                    continue_train=parameters['continue_train'],
                    directory=parameters['prefix'])
    State.init(sensor_number=parameters['sensor_number'],
               cell_size=cleaner.cell_limit)
    if not parameters['analysis']:    # 训练/测试模式下，需要初始化日志器，初始化环境
        Logger.init(console_log=parameters['console_log'],
                    file_log=parameters['file_log'],
                    directory=parameters['prefix'])
        env = Environment(train=parameters['train'],
                          continue_train=parameters['continue_train'],
                          sensor_number=parameters['sensor_number'],
                          worker_number=parameters['worker_number'],
                          max_episode=parameters['max_episode'],
                          batch_size=parameters['batch_size'],
                          epsilon_decay=parameters['epsilon_decay'],
                          learn_rate=parameters['learn_rate'],
                          gamma=parameters['gamma'],
                          detail=parameters['detail'],
                          cleaner=cleaner)
        env.start()
    else:
        alz = Analysis(train=parameters['train'])
        alz.start()

    Persistent.close()
