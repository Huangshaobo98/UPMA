import csv
import os
from typing import List
import numpy as np

from global_parameter import Global as g
import pandas as pd

class Persistent:
    __persistent_directory = None
    __train = None
    __continue_train = None
    __model_directory = None
    __data_directory = None
    __model_path = None
    __data_path = None
    __episode_data_path = None
    __npz_path = None

    # __network_model_directory = None
    # __network_model_path = None

    __file_handle = None  # 文件句柄，用于记录运行时状态
    __file_writer = None  # csv_writer
    __added_header = None  # 续训时应该已经创建好了header
    __episode_added_header = None
    __slot_log_save = False
    __trained_episode = 0
    __trained_epsilon = 1.0

    @staticmethod
    def init(train: bool,
             continue_train: bool,
             compare: bool,
             compare_method: str = "",
             directory="",
             suffix="",
             slot_log_save=False):

        if not directory == "":
            Persistent.__persistent_directory = directory

        Persistent.__train = train
        Persistent.__continue_train = continue_train

        Persistent.__added_header = continue_train  # 续训时应该已经创建好了header
        Persistent.__episode_added_header = continue_train

        Persistent.__model_directory = Persistent.__persistent_directory + "/model"
        Persistent.__data_directory = Persistent.__persistent_directory + ("/compare" if compare else ("/train" if train else "/test"))
        Persistent.__data_path = Persistent.__data_directory + ("/{}{}.csv".format(compare_method, suffix) if compare
                                                                else ("/running{}.csv".format(suffix) if train else "/Test{}.csv".format(suffix)))
        Persistent.__npz_path = Persistent.__data_directory + ('/{}{}.npz'.format(compare_method, suffix) if compare
                                                               else ("/running{}.npz".format(suffix) if train else "/Test{}.npz".format(suffix)))
        # if compare:
        #     Persistent.__data_path = Persistent.__data_directory + "/{}{}.csv".format(compare_method, suffix)
        # else:
        #     Persistent.__data_path = Persistent.__data_directory +

        # Persistent.__network_model_directory = Persistent.__data_directory + "/network_model"
        Persistent.__model_path = Persistent.__model_directory + "/model.h5"
        Persistent.__episode_data_path = Persistent.__data_directory + "/{}episode{}.csv"\
            .format(((compare_method + '_') if compare else ""), suffix)
        # if compare:
        #     Persistent.__episode_data_path = Persistent.__data_directory + "/{}_episode.csv".format(compare_method)
        # Persistent.__network_model_path = Persistent.__data_directory + ".npz"

        if not os.path.exists(Persistent.__model_directory):
            os.makedirs(Persistent.__model_directory)
        if not os.path.exists(Persistent.__data_directory):
            os.makedirs(Persistent.__data_directory)

        if train:
            with open(Persistent.__data_directory + '/start.txt', 'w') as f:
                pass
            if not continue_train:                      # 非断点续训练，要干掉原有的训练日志，从0开始
                if os.path.exists(Persistent.__model_path):
                    os.remove(Persistent.__model_path)
                if os.path.exists(Persistent.__data_path):
                    os.remove(Persistent.__data_path)
                if os.path.exists(Persistent.__episode_data_path):
                    os.remove(Persistent.__episode_data_path)
            else:
                assert os.path.exists(Persistent.__model_path)
                assert os.path.exists(Persistent.__data_path)
                assert os.path.exists(Persistent.__episode_data_path)

                with open(Persistent.__episode_data_path) as f:
                    try:
                        final_train_episode_info = f.readlines()[-1].split(',')
                        Persistent.__trained_episode = int(final_train_episode_info[0])
                        Persistent.__trained_epsilon = 0.5
                        Persistent.__episode_added_header = True
                    except ValueError:
                        Persistent.__episode_added_header = True
                    except IndexError:
                        Persistent.__episode_added_header = False
        else:
            if os.path.exists(Persistent.__data_path):
                os.remove(Persistent.__data_path)
            if os.path.exists(Persistent.__episode_data_path):
                os.remove(Persistent.__episode_data_path)
                # with open(Persistent.__data_path) as f:
                #     try:
                #         final_train_data_info = f.readlines()[-1].split(',')
                #         Persistent.__trained_epsilon = float(final_train_data_info[-1])
                #         Persistent.__added_header = True
                #     except ValueError:
                #         Persistent.__added_header = True
                #     except IndexError:
                #         Persistent.__added_header = False

        if slot_log_save:
            if not train and os.path.exists(Persistent.__data_path):
                os.remove(Persistent.__data_path)
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
        if not Persistent.__slot_log_save:
            return
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

    # @staticmethod
    # def save_network_model(cell_length, cell_limit, sensor_position):
    #     np.savez(Persistent.__network_model_directory, sensor_position=sensor_position)

    # @staticmethod
    # def load_network_model():
    #     file = np.load(Persistent.__network_model_path)
    #     return file['sensor_position']

    @staticmethod
    def npz_path() -> str:
        return Persistent.__npz_path

    @staticmethod
    def model_path() -> str:
        return Persistent.__model_path

    @staticmethod
    def data_path() -> str:
        return Persistent.__data_path

    @staticmethod
    def data_directory() -> str:
        return Persistent.__data_directory

    @staticmethod
    def trained_episode():
        return Persistent.__trained_episode

    @staticmethod
    def trained_epsilon():
        return Persistent.__trained_epsilon

    @staticmethod
    def model_directory():
        return Persistent.__model_directory

    @staticmethod
    def episode_data_path():
        return Persistent.__episode_data_path
