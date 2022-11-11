# 感觉也并不需要logging，简单实现一个能向控制台打印当前状态的日志器就行了
import os
import time


class Logger:
    def __init__(self, directory: str, console_log: bool):
        self.__console_log = console_log
        self.__log_directory = directory + "/log"
        log_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        self.__log_path = self.__log_directory + "/log_" + log_time + ".txt"
        if not os.path.exists(self.__log_directory):
            os.makedirs(self.__log_directory)
        self.__log_handle = open(self.__log_path, "w")

    def message(self, **kwargs):
        msg = ""
        for key, value in kwargs:
            msg += key
            msg += ": "
            msg += value
            msg += "; "
        msg += "\r\n"
        if self.__console_log:
            print(msg)
        self.__log_handle.write(msg)
