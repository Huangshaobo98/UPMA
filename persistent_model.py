import csv
import os
from global_parameter import Global as g


class Persistent:
    __persistent_directory = g.default_save_path

    def __init__(self, train: bool, continue_train: bool, directory=""):
        if not directory == "":
            Persistent.__persistent_directory = directory

        self.__model_directory = Persistent.__persistent_directory + "/model"
        self.__train_directory = Persistent.__persistent_directory + "/train"
        self.__test_directory = Persistent.__persistent_directory + "/test"

        self.__model_path = self.__model_directory + "/model.h5"
        self.__train_data_path = self.__train_directory + "/running.csv"
        self.__test_data_path = self.__test_directory + "/running.csv"

        if not os.path.exists(self.__model_directory):
            os.makedirs(self.__model_directory)
        if not os.path.exists(self.__train_directory):
            os.makedirs(self.__train_directory)
        if not os.path.exists(self.__test_directory):
            os.makedirs(self.__test_directory)

        self.__file_handle = None
        self.__file_writer = None
        self.__added_header = False

        if train:
            if not continue_train:                      # 非断点续训练，要干掉原有的训练日志，从0开始
                if os.path.exists(self.__model_path):
                    os.remove(self.__model_path)
                if os.path.exists(self.__train_data_path):
                    os.remove(self.__train_data_path)
            else:                                       # 断点训练，需要取出原有的训练模型，并且以追加形式训练模型
                self.__added_header = True
            self.__file_handle = open(self.__train_data_path, 'a+')
        else:
            # 测试模式，清空以往测试数据。此时模式不存在追加，只能从0开始
            if os.path.exists(self.__test_data_path):
                os.remove(self.__test_data_path)
            self.__file_handle = open(self.__test_data_path, 'w+')
        assert self.__file_handle is not None

        self.__file_writer = csv.writer(self.__file_handle)
        assert self.__file_writer is not None

    def save_data(self, datas):
        keys = []
        values = []
        for key, value in datas.items():
            keys.append(key)
            values.append(value)

        if not self.__added_header:
            self.__file_writer.writerow(keys)
            self.__added_header = True

        self.__file_writer.writerow(values)

    def model_path(self) -> str:
        return self.__model_path

    def train_log_path(self) -> str:
        return self.__train_data_path

    def test_log_path(self) -> str:
        return self.__test_data_path
