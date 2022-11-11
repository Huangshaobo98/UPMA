from environment_model import Environment
import sys

if __name__ == '__main__':

    commands = sys.argv[1:]
    console_enable = False
    train = True
    continue_train = False
    for item in commands:
        if item == "--console_log" or item == "log":
            console_enable = True
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

    env = Environment(console_enable, train, continue_train)
    env.start()
