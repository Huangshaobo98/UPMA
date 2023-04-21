import os
import signal
from main import Main
from multiprocessing import Process
from data.data_clean import DataCleaner
import time

workers = []
# cleaner = DataCleaner()
def signal_handler(p1, p2):
    for worker in workers:
        worker.terminate()
        # worker.is_alive()

def process(kwargs):
    print("Setting " + str(kwargs) + " begin on process id: " + str(os.getpid()) + "\n", end="")
    processor = Main(kwargs)
    processor.start()
    processor.end()

def learn_train_test():
    print("learn train")
    workers = []
    # for i in (0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.9},)))
    # for i in (0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.95},)))
    for i in (0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005):
        data = {'gamma': 0.85,
                'malicious': 0.2,
                'pho': 0.5,
                'learn_rate': i,
                'train': True}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))

    # for i in (0.001, 0.0001, 0.0005, 0.00005, 0.00001, 0.000005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.99, 'train': True},)))
    return workers

def gamma_train_test():
    print("gamma train")
    workers = []
    for i in (0.99, 0.95, 0.9, 0.75, 0.5, 0.3):
        data = {'gamma': i,
                'malicious': 0.2,
                'pho': 0.5,
                'learn_rate': 0.00005,
                'train': True}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def batch_size_train_test():
    print("batch_size train")
    workers = []
    for b in (32, 64, 128, 512, 1024):
        data = {'gamma': 0.85,
                'malicious': 0.2,
                'pho': 0.5,
                'learn_rate': 0.00005,
                'batch_size': b,
                'train': True}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def decay_train_test():
    print("ep decay train")
    workers = []
    for decay in (0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.999995, 0.999999):
        data = {'gamma': 0.85,
                'malicious': 0.2,
                'pho': 0.5,
                'learn_rate': 0.00005,
                'epsilon_decay': decay,
                'train': True}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def worker_train_test():
    print("worker_train_test")
    workers = []
    for i in (20, 200, 2000):
        workers.append(Process(target=process, args=({'worker_number': i, 'gamma': 0.9, 'learn_rate': 0.00005, 'train': True},)))
    return workers


def sensor_train_test():
    print("sensor_train_test")
    workers = []
    for i in (200, 500, 1000, 2000, 10000):
        workers.append(Process(target=process, args=({'sensor_number': i, 'gamma': 0.9, 'learn_rate': 0.00005, 'train': True},)))
    return workers

def sensor_worker_train():
    print("sensor_worker_train_test")
    workers = []
    for sensor in [500, 1000, 2000, 5000, 10000, 20000]:
        for worker in [0, 100, 200, 500, 1000, 2000, 5000, 10000]:
            data = {'sensor_number': sensor,
                    'worker_number': worker,
                    'gamma': 0.85,
                    'malicious': 0.2,
                    'pho': 0.5,
                    'learn_rate': 0.00005,
                    'max_episode': 500,
                    'train': True}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers
def reduce_rate_test():
    workers = []
    for mali in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for reduce_rate in [0.96875, 0.9375, 0.875, 0.75, 0.5]:
            data = {'sensor_number': 5000,
                    'worker_number': 500,
                    'gamma': 0.9,
                    'learn_rate': 0.00005,
                    'malicious': mali,
                    'pho': 0.5,
                    'train': False,
                    'reduce_rate': reduce_rate, 'suffix': '_reduce_{}_mali_{}_pho_0.5_new'.format(str(reduce_rate), str(mali))}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def mali_rate_test():
    workers = []
    for mali in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        data = {'sensor_number': 5000,
                'worker_number': 500,
                'gamma': 0.9,
                'learn_rate': 0.00005,
                'train': False,
                'malicious': mali,
                'suffix': '_mali_{}'.format(str(mali))}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    for mali in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        data = {'sensor_number': 5000,
                'worker_number': 500,
                'gamma': 0.9,
                'learn_rate': 0.00005,
                'train': False,
                'malicious': mali,
                'suffix': '_mali_{}_random'.format(str(mali)),
                'random_assignment': True,
                }
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def random_test():
    workers = []
    for mali in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        data = {'sensor_number': 5000,
                'worker_number': 500,
                'gamma': 0.9,
                'learn_rate': 0.00005,
                'malicious': mali,
                'pho': 0.5,
                'train': False,
                'reduce_rate': 0.875,
                'random_assignment': True,
                'suffix': '_reduce_0.875_mali_{}_pho_0.5_random'.format(str(mali))}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def compare_test():
    print("compare test")
    workers = []
    for worker in [5000, 2000, 1000, 500, 200, 100]:
        for sensor in [500, 1000, 5000, 10000]:
            for method in ['CCPP', 'RR', 'Greedy']: # RR Greedy
                data = {'sensor_number': sensor,
                        'worker_number': worker,
                        'gamma': 0.9,
                        'learn_rate': 0.00005,
                        'pho': 0.5,
                        'malicious': 0.0,
                        'reduce_rate': 0.875,
                        'train': False,
                        'compare': True,
                        'compare_method': method,
                        'suffix': '_mali_0_reduce_0.875_pho_0.5'}
                workers.append(Process(target=process,
                                       args=(data,),
                                       name=str(data)))
            data = {'sensor_number': sensor,
                    'worker_number': worker,
                    'gamma': 0.9,
                    'learn_rate': 0.00005,
                    'pho': 0.5,
                    'malicious': 0.0,
                    'reduce_rate': 0.875,
                    'train': False,
                    'suffix': '_mali_0_reduce_0.875_pho_0.5'}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def test_test():
    print("compare test")
    workers = []
    for worker in [0]:
        for sensor in [200, 500, 1000, 2000, 5000, 10000]:
            data = {'sensor_number': sensor,
                    'worker_number': worker,
                    'gamma': 0.9,
                    'learn_rate': 0.00005,
                    'train': False}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def worker_test_compare():
    workers = []
    for i in [0, 50, 500, 5000]:
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'worker_number': i, 'train': False,
                      'compare': False, 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'worker_number': i, 'train': False,
                      'compare': False, 'compare_method': 'RR', 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'worker_number': i, 'train': False,
                      'compare': False, 'compare_method': 'Greedy', 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'worker_number': i, 'train': False,
                      'compare': False, 'compare_method': 'CCPP', 'console_log': False})))
    return workers

def sensor_test_compare():
    workers = []
    for i in [500, 1000, 5000, 10000]:
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'sensor_number': i, 'train': False,
                      'compare': False, 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'sensor_number': i, 'train': False,
                      'compare': False, 'compare_method': 'RR', 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'sensor_number': i, 'train': False,
                      'compare': False, 'compare_method': 'Greedy', 'console_log': False})))
        workers.append(Process(target=process({'learn_rate': 0.00005, 'gamma': 0.9,
                      'sensor_number': i, 'train': False,
                      'compare': False, 'compare_method': 'CCPP', 'console_log': False})))
        return workers

# def pho_test():
#     workers = []
#     for seed in range(400):
#         for pho in [0.3, 0.5, 0.7, 0.9]:
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9,
#                           'worker_number': 5000, 'train': False, 'malicious': 0.5,
#                           'console_log': False, 'seed': seed, 'pho': pho,
#                           'suffix': '_' + str(seed) + '_pho_' + str(pho)},)))
#     return workers

def pho_test():
    workers = []
    for mali in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for pho in [0.3, 0.5, 0.7, 0.9]:
            data = {'sensor_number': 5000,
                    'worker_number': 500,
                    'gamma': 0.9,
                    'learn_rate': 0.00005,
                    'train': False,
                    'malicious': mali,
                    'pho': pho,
                    'suffix': '_pho_{}_mali_{}_new'.format(str(pho), str(mali)),
                    }
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def win_test():
    workers = []
    for seed in range(50, 100):
        for win in [15]:
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9,
                          'worker_number': 100, 'train': False, 'malicious': 0.5, 'pho': 0.9,
                          'console_log': False, 'seed': seed, 'windows_length': win,
                          'suffix': '_' + str(seed) + '_win_' + str(win)},)))
    return workers

def worker_performance():
    workers = []
    for seed in range(120):
        for malicious in [0.2, 0.4, 0.6, 0.8]:
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9,
                          'worker_number': 5000, 'train': False, 'malicious': malicious,
                          'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
                          'suffix': '_' + str(seed) + '_mali_' + str(malicious)},)))
    return workers

def worker_performance_random():
    workers = []
    for seed in range(120):
        for malicious in [0.2, 0.4, 0.6, 0.8]:
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9,
                          'worker_number': 5000, 'train': False, 'malicious': malicious,
                          'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': True,
                          'suffix': '_' + str(seed) + '_mali_' + str(malicious) + '_random'},)))
    return workers

# def worker_repeat_train():
#     workers = []
#     for worker_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'worker_number': worker_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'CCPP',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for worker_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'worker_number': worker_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'Greedy',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for worker_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'worker_number': worker_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'RR',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for worker_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'worker_number': worker_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     return workers
#
# def sensor_repeat_train():
#     workers = []
#     for sensor_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'sensor_number': sensor_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'CCPP',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for sensor_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'sensor_number': sensor_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'Greedy',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for sensor_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'sensor_number': sensor_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'compare': True, 'compare_method': 'RR',
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#     for sensor_number in [500, 1000, 5000, 10000]:
#         for seed in range(120):
#             workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#                                                           'sensor_number': sensor_number, 'train': False,
#                                                           'malicious': 0.1,
#                                                           'console_log': False, 'seed': seed, 'pho': 0.7,
#                                                           'random_assignment': False,
#                                                           'suffix': '_' + str(seed)},)))
#             # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#             #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
#             #               'compare': True, 'compare_method': 'CCPP',
#             #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
#             #               'suffix': '_' + str(seed)},)))
#             # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#             #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
#             #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
#             #               'compare': True, 'compare_method': 'Greedy',
#             #               'suffix': '_' + str(seed)},)))
#             # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
#             #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
#             #               'compare': True, 'compare_method': 'RR',
#             #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
#             #               'suffix': '_' + str(seed)},)))
#     return workers

# def cell_6_train_test():
#     print("sensor_worker_train_test")
#     workers = []
#     cleaner = DataCleaner(
#                  x_limit=6,
#                  y_limit=6,
#                  x_range=[0, 2000],  # 可以是角度，也可以是距离
#                  y_range=[0, 1800],
#                  range_is_angle=False, # 很重要的参数，确定是角度的话需要乘以地球系数
#                  Norm=True,
#                  Norm_centers=[[500, 600], [1400, 1100]],
#                  Norm_centers_ratio=[0.4, 0.6], # 每一个分布中心所占比率
#                  Norm_sigma=[1, 1],       # 正态分布的方差
#                  Norm_gain=[400, 600],        # 正态分布系数，用于控制器辐射半径大小
#                  No_data_set_need=True)
#     # for gamma in [0.3, 0.5, 0.75, 0.95, 0.99]:
#     #     data = {'sensor_number': 1000,
#     #             'worker_number': 0,
#     #             'console_log': False,
#     #             'gamma': gamma,
#     #             'learn_rate': 1e-4,
#     #             'uav_energy': 10,
#     #             'max_episode': 500,
#     #             'train': True,
#     #             'cleaner': cleaner}
#     #     workers.append(Process(target=process,
#     #                            args=(data,),
#     #                            name=str(data)))
#     for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
#         data = {'sensor_number': 1000,
#                 'worker_number': 0,
#                 'console_log': False,
#                 'gamma': 0.9,
#                 'learn_rate': lr,
#                 'uav_energy': 10,
#                 'max_episode': 500,
#                 'train': True,
#                 'cleaner': cleaner}
#         workers.append(Process(target=process,
#                                args=(data,),
#                                name=str(data)))
#     return workers
#
# def cell_6_train_test_1():
#     print("sensor_worker_train_test")
#     workers = []
#     cleaner = DataCleaner(
#                  x_limit=6,
#                  y_limit=6,
#                  x_range=[0, 2000],  # 可以是角度，也可以是距离
#                  y_range=[0, 1800],
#                  range_is_angle=False, # 很重要的参数，确定是角度的话需要乘以地球系数
#                  Norm=False,
#                  # Norm_centers=[[500, 600], [1400, 1100]],
#                  # Norm_centers_ratio=[0.4, 0.6], # 每一个分布中心所占比率
#                  # Norm_sigma=[1, 1],       # 正态分布的方差
#                  # Norm_gain=[400, 600],        # 正态分布系数，用于控制器辐射半径大小
#                  No_data_set_need=True)
#     for sen_num in [250, 500, 750, 1000]:
#         data = {'sensor_number': sen_num,
#                 'worker_number': 0,
#                 'console_log': False,
#                 'gamma': 0.9,
#                 'learn_rate': 1e-3,
#                 'uav_energy': 10,
#                 'max_episode': 1000,
#                 'train': True,
#                 'cleaner': cleaner}
#         workers.append(Process(target=process,
#                                args=(data,),
#                                name=str(data)))
#     # for lr in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:
#     #     data = {'sensor_number': 1000,
#     #             'worker_number': 0,
#     #             'console_log': False,
#     #             'gamma': 0.9,
#     #             'learn_rate': lr,
#     #             'uav_energy': 10,
#     #             'max_episode': 500,
#     #             'train': True,
#     #             'cleaner': cleaner}
#     #     workers.append(Process(target=process,
#     #                            args=(data,),
#     #                            name=str(data)))
#     return workers

def cell_10_t_drive_lr_train():
    print("cell_10_lr_train")
    workers = []
    cleaner = DataCleaner()
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        data = {'sensor_number': 5000,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': lr,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': True,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_gamma_train():
    print("cell_10_lr_train")
    workers = []
    cleaner = DataCleaner()
    for gamma in [0.5, 0.75, 0.95, 0.99]:
        data = {'sensor_number': 5000,
                'worker_number': 1000,
                'console_log': False,
                'gamma': gamma,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': True,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_work_num_train():
    print("cell_10_worknum_train")
    workers = []
    cleaner = DataCleaner()
    for worker_num in [250, 500, 2500, 5000, 10000]:
        data = {'sensor_number': 5000,
                'worker_number': worker_num,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': True,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_cost_train():
    print("cell_10_cost_train")
    workers = []
    cleaner = DataCleaner()
    for cost in [500, 1000, 2000, 4000, 8000]:
        data = {'sensor_number': 5000,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': cost,
                'train': True,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_cost_test():
    print("cell_10_cost_test")
    workers = []
    cleaner = DataCleaner()
    for cost in [250]: # 500, 1000, 2000, 4000, 8000
        data = {'sensor_number': 5000,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': cost,
                'train': False,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_work_num_test():
    print("cell_10_worknum_test")
    workers = []
    cleaner = DataCleaner()
    for worker_num in [250, 500, 2500, 5000, 10000]:
        data = {'sensor_number': 5000,
                'worker_number': worker_num,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': False,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_compare_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    for compare_method in ['RR', 'CCPP', 'Greedy']:
        data = {'sensor_number': 5000,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': False,
                'compare': True,
                'compare_method': compare_method,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_cpu_aoi_compare_worker_num_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    for worker_number in [250, 500, 750, 1000, 1250, 1500]:
        for task_assignment_policy in ['g-greedy']: # 'greedy', 'g-greedy', 'random', 'genetic'
            data = {'sensor_number': 5000,
                    'worker_number': worker_number,
                    'console_log': False,
                    'gamma': 0,
                    'learn_rate': 0,
                    'uav_energy': 77,
                    'max_episode': 0,
                    'cost_limit': 2000,
                    'task_assignment_policy': task_assignment_policy,
                    'train': False,
                    'compare': True,
                    'compare_method': 'RR',
                    'suffix': 'nop',
                    'cleaner': cleaner}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def cell_10_t_drive_cpu_aoi_compare_sensor_number_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    for sensor_number in [1000, 2000, 3000, 4000, 5000, 6000]:
        for task_assignment_policy in ['g-greedy']:
            data = {'sensor_number': sensor_number,
                    'worker_number': 1000,
                    'console_log': False,
                    'gamma': 0,
                    'learn_rate': 0,
                    'uav_energy': 77,
                    'max_episode': 0,
                    'cost_limit': 2000,
                    'task_assignment_policy': task_assignment_policy,
                    'train': False,
                    'compare': True,
                    'compare_method': 'RR',
                    'suffix': 'nop',
                    'cleaner': cleaner}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def cell_10_t_drive_cpu_aoi_compare_cost_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    # have_done = [500, 1000, 2000]
    for cost in [500, 1000, 1500, 2000, 2500, 3000]:
        for task_assignment_policy in ['g-greedy']:
            # if cost in have_done and task_assignment_policy in ['greedy', 'g-greedy', 'random']:
            #     continue
            data = {'sensor_number': 5000,
                    'worker_number': 1000,
                    'console_log': False,
                    'gamma': 0,
                    'learn_rate': 0,
                    'uav_energy': 77,
                    'max_episode': 0,
                    'cost_limit': cost,
                    'task_assignment_policy': task_assignment_policy,
                    'train': False,
                    'compare': True,
                    'compare_method': 'RR',
                    'suffix': 'nop',
                    'cleaner': cleaner}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def cell_10_t_drive_union_method_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    methods = ['RR', 'GCTA', 'gGreedy', 'Random', 'D3QN_GCTA', 'D3QN_gGreedy', 'D3QN_Random', 'RR_GCTA', 'RR_gGreedy', 'RR_Random', 'Greedy_GCTA', 'Greedy_gGreedy', 'Greedy_Random']
    tsk_mtd = ['greedy', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random']    # 空表示随意，反正也不会用上
    wkr_num = [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    no_uav = [False, True, True, True, False, False, False, False, False, False, False, False, False]
    cmp = [True, False, False, False, False, False, False, True, True, True, True, True, True]
    cmp_mth = ['RR', '', '', '', '', '', '', 'RR', 'RR', 'RR', 'Greedy', 'Greedy', 'Greedy']
    over_state = [True, True, True, True, False, False, False, True, True, True, True, True, True]
    for cnt in range(13):
        if over_state[cnt]:
            continue
        data = {'sensor_number': 5000,
                'worker_number': wkr_num[cnt],
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'task_assignment_policy': tsk_mtd[cnt],
                'train': False,
                'compare': cmp[cnt],
                'compare_method': cmp_mth[cnt],
                'no_uav': no_uav[cnt],
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_sensor_num_train():
    print("cell_10_cost_train")
    workers = []
    cleaner = DataCleaner()
    for sennum in [1000, 2000, 3000, 4000, 6000]:
        data = {'sensor_number': sennum,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': True,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

def cell_10_t_drive_sensor_num_test():
    print("cell_10_cost_train")
    workers = []
    cleaner = DataCleaner()
    for sennum in [1000, 2000, 3000, 4000, 6000]:
        data = {'sensor_number': sennum,
                'worker_number': 1000,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'train': False,
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers


def cell_10_t_drive_pho_test():
    print("cell_10_cost_train")
    workers = []
    cleaner = DataCleaner()
    for mali in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for pho in [0.3, 0.7]:
            data = {'sensor_number': 5000,
                    'worker_number': 1000,
                    'console_log': False,
                    'gamma': 0.9,
                    'learn_rate': 0.001,
                    'uav_energy': 77,
                    'max_episode': 600,
                    'cost_limit': 2000,
                    'malicious': mali,
                    'pho': pho,
                    'suffix': 'pho_{}_mali_{}'.format(str(pho), str(mali)),
                    'train': False,
                    # 'compare': True,
                    # 'compare_method': 'RR',
                    'cleaner': cleaner}
            workers.append(Process(target=process,
                                   args=(data,),
                                   name=str(data)))
    return workers

def cell_10_t_drive_mali_tsk_compare_test():
    print("cell_10_compare_test")
    workers = []
    cleaner = DataCleaner()
    methods = ['GCTA', 'gGreedy', 'Random', 'genetic' 'D3QN_GCTA', 'D3QN_gGreedy', 'D3QN_Random', 'RR_GCTA', 'RR_gGreedy', 'RR_Random', 'Greedy_GCTA', 'Greedy_gGreedy', 'Greedy_Random']
    tsk_mtd = ['greedy',                                     # RR
               'greedy', 'g-greedy', 'random', 'genetic',   # 4tsk
               'greedy', 'g-greedy', 'random', 'genetic',   # d3qn + 4tsk
               'greedy', 'g-greedy', 'random', 'genetic',   # rr + 4tsk
               'greedy', 'g-greedy', 'random', 'genetic']   # greedy + 4tsk
    # wkr_num = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    no_uav = [False,                                        # RR
              True, True, True, True,                       # 4tsk  # 无无人机默认d3qn
              False, False, False, False,                   # d3qn + 4tsk
              False, False, False, False,                   # rr + 4tsk
              False, False, False, False]                   # greedy + 4tsk
    cmp = [True,
           False, False, False, False,
           False, False, False, False,
           True, True, True, True,
           True, True, True, True,]
    cmp_mth = ['RR',
               '', '', '', '',
               '', '', '', '',
               'RR', 'RR', 'RR', 'RR',
               'Greedy', 'Greedy', 'Greedy', 'Greedy']
    not_done = [4, 8, 12, 16]
    for cnt in range(17):
        data = {'sensor_number': 5000,
                'worker_number': 0,
                'console_log': False,
                'gamma': 0.9,
                'learn_rate': 0.001,
                'uav_energy': 77,
                'max_episode': 600,
                'cost_limit': 2000,
                'task_assignment_policy': tsk_mtd[cnt],
                'train': False,
                'compare': cmp[cnt],
                'compare_method': cmp_mth[cnt],
                'no_uav': no_uav[cnt],
                'suffix': '326',
                'cleaner': cleaner}
        workers.append(Process(target=process,
                               args=(data,),
                               name=str(data)))
    return workers

if __name__ == '__main__':

    signal.signal(signal.SIGTERM, signal_handler)
    print("master process: " + str(os.getpid()) + "\n", end="")

    # workers.extend(reduce_rate_test())
    # workers.extend(random_test())
    # workers.extend(sensor_worker_train())

    workers.extend(cell_10_t_drive_mali_tsk_compare_test())
    # workers.extend(learn_train_test()) # 7
    # workers.extend(gamma_train_test()) # 6
    # workers.extend(gamma_train_test())
    # workers.extend(batch_size_train_test()) # 5
    # workers.extend(decay_train_test()) # 8
    # workers.extend(worker_performance())
    for worker in workers:
        worker.daemon = True

    pool_number = 15

    running = []
    wait_for_running = workers.copy()

    while len(running) > 0 or len(wait_for_running) > 0:
        # 向池中添加worker
        while len(wait_for_running) > 0 and len(running) < pool_number:
            worker = wait_for_running.pop(0)
            worker.start()
            time.sleep(1)
            print(worker.name + ' has run in process {}\n'.format(worker.pid), end='')
            running.append(worker)

        # 回收zombie进程
        repeat = True
        while repeat:
            repeat = False
            for worker in running:
                if not worker.is_alive():
                    worker.join()
                    running.remove(worker)
                    repeat = True
                    break
        # 每60s判断一次，向池中添加worker，或者回收zombie
        time.sleep(30)

