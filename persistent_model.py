import csv
import os

import numpy as np

from global_parameter import Global as g
import pandas as pd


class Persistent:
    __persistent_directory = g.default_save_path
    __train = g.default_train
    __continue_train = g.default_continue_train
    __model_directory = g.default_save_path + "/model"
    __data_directory = g.default_save_path + ("/train" if g.default_train else "/test")
    __model_path = __model_directory + "/model.h5"
    __data_path = __data_directory + "/running.csv"
    __episode_data_path = __data_directory + "/episode.csv"

    __network_model_directory = __data_path + "/network_model"
    __network_model_path = __network_model_directory + ".npz"

    __file_handle = None  # 文件句柄，用于记录运行时状态
    __file_writer = None  # csv_writer
    __added_header = g.default_continue_train  # 续训时应该已经创建好了header
    __episode_added_header = g.default_continue_train

    @staticmethod
    def init(analysis: bool,
             train: bool,
             continue_train: bool,
             directory=""):

        if not directory == "":
            Persistent.__persistent_directory = directory

        Persistent.__train = train
        Persistent.__continue_train = continue_train

        Persistent.__added_header = continue_train  # 续训时应该已经创建好了header
        Persistent.__episode_added_header = continue_train

        Persistent.__model_directory = Persistent.__persistent_directory + "/model"
        Persistent.__data_directory = Persistent.__persistent_directory + ("/train" if train else "/test")
        Persistent.__network_model_directory = Persistent.__data_directory + "/network_model"

        Persistent.__model_path = Persistent.__model_directory + "/model.h5"
        Persistent.__data_path = Persistent.__data_directory + "/running.csv"
        Persistent.__episode_data_path = Persistent.__data_directory + "/episode.csv"
        Persistent.__network_model_path = Persistent.__data_directory + ".npz"

        if not os.path.exists(Persistent.__model_directory):
            os.makedirs(Persistent.__model_directory)
        if not os.path.exists(Persistent.__data_directory):
            os.makedirs(Persistent.__data_directory)

        if not analysis:    # 训练/测试模式
            if not continue_train:                      # 非断点续训练，要干掉原有的训练日志，从0开始
                if os.path.exists(Persistent.__model_path):
                    os.remove(Persistent.__model_path)
                if os.path.exists(Persistent.__data_path):
                    os.remove(Persistent.__data_path)
                if os.path.exists(Persistent.__episode_data_path):
                    os.remove(Persistent.__episode_data_path)

            Persistent.__file_handle = open(Persistent.__data_path, 'a+')   # 追加模式，方便后面存储数据
            assert Persistent.__file_handle is not None

            Persistent.__file_writer = csv.writer(Persistent.__file_handle)
            assert Persistent.__file_writer is not None

    @staticmethod
    def close():
        if Persistent.__file_handle is not None:
            Persistent.__file_handle.close()

    # 仅在分析数据时会调用到此方法
    @staticmethod
    def read_data():
        if not os.path.exists(Persistent.__data_path):
            raise FileNotFoundError("Data path: {} not exist".format(Persistent.__data_path))

        slot_data = pd.read_csv(Persistent.__data_path)
        assert slot_data is not None
        return slot_data

    @staticmethod
    def read_episode_data():
        if not os.path.exists(Persistent.__episode_data_path):
            raise FileNotFoundError("Episode data path: {} not exist".format(Persistent.__episode_data_path))

        episode_data = pd.read_csv(Persistent.__episode_data_path)
        assert episode_data is not None
        return episode_data

    # save a row data
    @staticmethod
    def save_data(datas):
        keys = []
        values = []
        for key, value in datas.items():
            keys.append(key)
            values.append(value)

        if not Persistent.__added_header:
            Persistent.__file_writer.writerow(keys)
            Persistent.__added_header = True

        Persistent.__file_writer.writerow(values)

    @staticmethod
    def save_episode_data(datas):
        keys = []
        values = []
        for key, value in datas.items():
            keys.append(key)
            values.append(value)

        with open(Persistent.__episode_data_path, 'a+') as fd:
            csv_writer = csv.writer(fd)
            if not Persistent.__episode_added_header:
                csv_writer.writerow(keys)
                Persistent.__episode_added_header = True
            csv_writer.writerow(values)

    @staticmethod
    def save_network_model(cell_length, cell_limit, sensor_position):
        np.savez(Persistent.__network_model_directory,
                cell_length=np.array(cell_length),
                cell_limit=np.array(cell_limit),
                sensor_position=sensor_position)

    @staticmethod
    def load_network_model():
        npfile = np.load(Persistent.__network_model_path)
        return npfile['cell_length'][0], npfile['cell_limit'][0], npfile['sensor_position']

    @staticmethod
    def model_path() -> str:
        return Persistent.__model_path

    @staticmethod
    def data_path() -> str:
        return Persistent.__data_path
