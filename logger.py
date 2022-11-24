# 感觉也并不需要logging，简单实现一个能向控制台打印当前状态的日志器就行了
import os
import datetime
from global_parameter import Global as g

class Logger:
    __console_log = g.default_console_log
    __file_log = g.default_file_log
    __log_directory = g.default_save_path + "/log"
    __log_path = __log_directory + "/log_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ".txt"
    __log_handle = None

    @staticmethod
    def init(console_log: bool, file_log: bool, directory=""):
        if not directory == "": # 用户自定义日志存储目录
            Logger.__log_directory = directory + "/log"
            Logger.__log_path = Logger.__log_directory + "/log_" \
                                + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ".txt"

        if not os.path.exists(Logger.__log_directory):
            os.makedirs(Logger.__log_directory)

        Logger.__console_log = console_log
        Logger.__file_log = file_log
        Logger.__log_handle = open(Logger.__log_path, "w") if Logger.__file_log else None
        assert (Logger.__file_log and Logger.__log_handle is not None) or (not Logger.__file_log)

    @staticmethod
    def file_log(msg: str):
        if Logger.__file_log:
            Logger.__log_handle.write(msg)

    @staticmethod
    def console_log(msg: str):
        if Logger.__console_log:
            print(msg)

    @staticmethod
    def log(msg: str):
        Logger.file_log(msg + "\n")
        Logger.console_log(msg)

    @staticmethod
    def log_time():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
