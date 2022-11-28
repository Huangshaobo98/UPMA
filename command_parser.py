import sys
from global_parameter import Global as g


def command_parse():
    commands = sys.argv[1:]
    console_log = g.default_console_log
    file_log = g.default_file_log
    train = g.default_train
    continue_train = g.default_continue_train
    analysis = g.default_analysis
    set_train = False
    prefix = ""
    for item in commands:
        command = item.lower()
        if command in ("--analysis", "--analysis-true", "-a", "-at"):
            analysis = True
        elif command in ("--analysis-false", "-af"):
            analysis = False
        elif command in ("--console_log", "--console_log-true", "-c", "-ct"):
            console_log = True
        elif command in ("--console_log-false", "-c", "-cf"):
            console_log = False
        elif command in ("--file_log", "--file_log-true", "-f", "-ft"):
            file_log = True
        elif command in ("--file_log-false", "-ff"):
            file_log = False
        elif not set_train and command in ("--train", "--train-true"):
            train = True
            set_train = True
        elif not set_train and command in ("--test", "--test-true"):
            train = False
            set_train = True
        elif set_train and command in ("--test", "--test-true", "--train", "--train-true"):
            raise ValueError("Command error, can not set train/test mode at the same time")
        elif command in ("--continue_train", "--continue_train-true"):
            continue_train = True
        elif command in ("--continue_train-false"):
            continue_train = False
        else:
            raise ValueError("Command error, try ./main.py or ./main.py --train or ./main.py -h for command help")
    return analysis, console_log, file_log, train, continue_train, prefix
