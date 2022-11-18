import sys
from global_parameter import Global as g


def command_parse():
    commands = sys.argv[1:]
    console_log = g.default_console_log
    file_log = g.default_file_log
    train = g.default_train
    continue_train = g.default_continue_train
    for item in commands:
        if item == "--console_log":
            console_log = True
            continue
        if item == "--file_log":
            file_log = True
            continue
        if item == "--train":
            train = True
            continue
        if item == "--test":
            train = False
            continue
        if item == "--continue_train":
            continue_train = True
            continue
        raise SyntaxError("Command error, try ./main.py or ./main.py --train")
    return console_log, file_log, train, continue_train
