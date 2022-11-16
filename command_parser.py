import sys


def command_parse():
    commands = sys.argv[1:]
    console_log = False
    file_log = False
    train = True
    continue_train = False
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
