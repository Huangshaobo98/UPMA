import os
import signal
from main import Main
from multiprocessing import Process
from data.data_clean import DataCleaner
import time

workers = []
cleaner = DataCleaner()
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
    for i in (0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005):
        workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.9, 'train': True},)))

    # for i in (0.001, 0.0001, 0.0005, 0.00005, 0.00001, 0.000005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.99, 'train': True},)))
    return workers

def gamma_train_test():
    print("gamma train")
    workers = []
    for i in (0.99, 0.95, 0.75, 0.5, 0.3):
        workers.append(Process(target=process, args=({'gamma': i, 'learn_rate': 0.00005, 'train': True},)))
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
    for worker in [10000]:
        for sensor in [200, 500, 1000, 2000, 5000, 10000]:
            for gamma in [0.8, 0.9]:
                data = {'sensor_number': sensor,
                        'worker_number': worker,
                        'gamma': gamma,
                        'learn_rate': 0.00005,
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

def worker_repeat_train():
    workers = []
    for worker_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'worker_number': worker_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'CCPP',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for worker_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'worker_number': worker_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'Greedy',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for worker_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'worker_number': worker_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'RR',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for worker_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'worker_number': worker_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    return workers

def sensor_repeat_train():
    workers = []
    for sensor_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'sensor_number': sensor_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'CCPP',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for sensor_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'sensor_number': sensor_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'Greedy',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for sensor_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'sensor_number': sensor_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'compare': True, 'compare_method': 'RR',
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
    for sensor_number in [500, 1000, 5000, 10000]:
        for seed in range(120):
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
                                                          'sensor_number': sensor_number, 'train': False,
                                                          'malicious': 0.1,
                                                          'console_log': False, 'seed': seed, 'pho': 0.7,
                                                          'random_assignment': False,
                                                          'suffix': '_' + str(seed)},)))
            # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
            #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
            #               'compare': True, 'compare_method': 'CCPP',
            #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
            #               'suffix': '_' + str(seed)},)))
            # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
            #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
            #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
            #               'compare': True, 'compare_method': 'Greedy',
            #               'suffix': '_' + str(seed)},)))
            # workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9, 'cleaner': cleaner,
            #               'sensor_number': sensor_number, 'train': False, 'malicious': 0.1,
            #               'compare': True, 'compare_method': 'RR',
            #               'console_log': False, 'seed': seed, 'pho': 0.7, 'random_assignment': False,
            #               'suffix': '_' + str(seed)},)))
    return workers

if __name__ == '__main__':

    signal.signal(signal.SIGTERM, signal_handler)
    print("master process: " + str(os.getpid()) + "\n", end="")

    # workers.extend(reduce_rate_test())
    # workers.extend(random_test())
    workers.extend(pho_test())
    workers.extend(reduce_rate_test())
    # workers.extend(mali_rate_test())
    # workers.extend(gamma_train_test())
    # workers.extend(worker_repeat_train())
    # workers.extend(gamma_train_test())
    # workers.extend(worker_performance())
    for worker in workers:
        worker.daemon = True

    pool_number = 36

    running = []
    wait_for_running = workers.copy()

    while len(running) > 0 or len(wait_for_running) > 0:
        # 向池中添加worker
        while len(wait_for_running) > 0 and len(running) < pool_number:
            worker = wait_for_running.pop(0)
            worker.start()
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

