from global_parameter import Global as g
import os


def command_parse(commands,
                  kwargs: dict = {}):
    command_iter = iter(commands)
    parameters = {
        'console_log': g.default_console_log,
        'file_log': g.default_file_log,
        'train': g.default_train,
        'continue_train': g.default_continue_train,
        'analysis': g.default_analysis,
        'prefix': "",
        # 'cell_limit': g.default_cell_limit,           # 边界大小
        'sensor_number': g.default_sensor_number,     # 传感器数量
        'worker_number': g.default_worker_number,       # worker数量
        'max_episode': g.default_max_episode,      # 最大训练episode
        # 'max_slot': g.default_max_slot,         # 最大时隙
        'batch_size': g.default_batch_size,       # 批大小
        'learn_rate': g.default_learn_rate,     # 学习率
        'gamma': g.default_gamma,                # 折扣系数
        'epsilon_decay': g.default_epsilon_decay,     # 探索率衰减
        'detail': g.default_detail_log
    }
    for key, value in kwargs.items():
        parameters[key] = value

    set_train = False
    while True:
        try:
            item = next(command_iter)
        except StopIteration:
            break
        command = item.lower()
        if command in ("--sensor_number", "-sensor"):
            parameters['sensor_number'] = int(next(command_iter))
            assert parameters['sensor_number'] > 0
        # elif command in ("--cell_limit", "-cell"):
        #     parameters['cell_limit'] = int(next(command_iter))
        #     assert parameters['cell_limit'] > 0
        elif command in ("--worker_number", "-worker"):
            parameters['worker_number'] = int(next(command_iter))
            assert parameters['worker_number'] >= 0
        elif command in ("--max_episode", "-episode"):
            parameters['max_episode'] = int(next(command_iter))
            assert parameters['max_episode'] > 0
        # elif command in ("--max_slot", "-slot"):
        #     parameters['max_slot'] = int(next(command_iter))
        #     assert parameters['max_slot'] > 0
        elif command in ("--batch_size", "-batch"):
            parameters['batch_size'] = int(next(command_iter))
            assert parameters['batch_size'] > 0
        elif command in ("--learn_rate", "-lr"):
            parameters['learn_rate'] = float(next(command_iter))
            assert 0 < parameters['learn_rate'] < 1
        elif command in ("--gamma", "-gamma"):
            parameters['gamma'] = float(next(command_iter))
            assert 0 < parameters['gamma'] < 1
        elif command in ("--epsilon_decay", "-decay"):
            parameters['epsilon_decay'] = float(next(command_iter))
            assert 0 < parameters['epsilon_decay'] < 1
        elif command in ("--prefix", "-p"):
            parameters['prefix'] = str(next(command_iter))
            assert len(parameters['prefix']) > 0
        elif command in ("--detail_log", "-detail"):
            parameters['detail'] = True
            assert type(parameters['detail']) is bool
        elif command in ("--analysis", "-a", "-analysis"):
            parameters['analysis'] = True
            assert type(parameters['analysis']) is bool
        elif command in ("--console", "--console_log", "-console"):
            parameters['console_log'] = True
            assert type(parameters['console_log']) is bool
        elif command in ("--file", "--file_log", "-file"):
            parameters['file_log'] = True
            assert type(parameters['file_log']) is bool
        elif command in ("--test", "--train", "-train", "-test"):
            if set_train:
                raise ValueError("Command error, can not set train/test mode at the same time")
            if command in ("--test", "-test"):
                parameters['train'] = False
            if command in ("--train", "-train"):
                parameters['train'] = True
            assert type(parameters['train']) is bool
            set_train = True
        elif command in ("--continue_train", "-continue"):
            parameters['continue_train'] = True
            assert type(parameters['continue_train']) is bool
        else:
            raise ValueError("Command error at \"{}\", try ./main.py or ./main.py --train or ./main.py".format(command))

    if parameters['prefix'] == "":
        parameters['prefix'] = os.getcwd() + "/save/t-drive" \
                                + "_sen_" + str(parameters['sensor_number']) \
                                + "_wkr_" + str(parameters['worker_number']) \
                                + "_epi_" + str(parameters['max_episode']) \
                                + "_bat_" + str(parameters['batch_size']) \
                                + "_lr_" + str(parameters['learn_rate']) \
                                + "_gama_" + str(parameters['gamma']) \
                                + "_epd_" + str(parameters['epsilon_decay'])
    return parameters
