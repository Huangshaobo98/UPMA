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
    def __init__(self, kwargs: dict = {}):
        tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
        parameters = command_parse(sys.argv[1:], kwargs)
        cleaner = DataCleaner()
        # self.analysis = parameters['analysis']
        Persistent.init(train=parameters['train'],
                        continue_train=parameters['continue_train'],
                        directory=parameters['prefix'])
        State.init(sensor_number=parameters['sensor_number'],
                   cell_size=cleaner.cell_limit)
        Logger.init(console_log=parameters['console_log'],
                    file_log=parameters['file_log'],
                    directory=parameters['prefix'])
        self.env = Environment(train=parameters['train'],
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

    def start(self):
        self.env.start()

    def end(self):
        Persistent.close()


if __name__ == '__main__':
    processor = Main({'worker_number': 10000})
    processor.start()
    processor.end()

