from environment_model import Environment
from command_parser import command_parse
import sys

if __name__ == '__main__':

    commands = sys.argv[1:]

    console_log, file_log, train, continue_train = command_parse()
    env = Environment(console_log, file_log, train, continue_train)
    env.start()
