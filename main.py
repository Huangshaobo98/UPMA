from environment_model import Environment
from command_parser import command_parse
from global_parameter import Global
from persistent_model import Persistent
from logger import Logger
from analysis import Analysis
import sys

if __name__ == '__main__':

    Global.init()
    commands = sys.argv[1:]
    analysis, console_log, file_log, train, continue_train, directory = command_parse()
    Persistent.init(analysis, train, continue_train, directory)
    if not analysis:    # 训练/测试模式下，需要初始化日志器，初始化环境
        Logger.init(console_log, file_log)
        env = Environment(train, continue_train)
        env.start()
    else:
        alz = Analysis(train)
        alz.start()

    Persistent.close()
