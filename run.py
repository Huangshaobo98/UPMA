import os
import signal
from main import Main
from multiprocessing import Process
from data_clean import DataCleaner

workers = []
cleaner = DataCleaner()
def signal_handler():
    for worker in workers:
        worker.terminate()

def process(kwargs):
    print("Setting " + str(kwargs) + " begin on process id: " + str(os.getpid()) + "\n", end="")
    processor = Main(kwargs)
    processor.start()
    processor.end()

def learn_train_test():
    workers = []
    # for i in (0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.9},)))
    # for i in (0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
    #     workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.95},)))
    for i in (0.001, 0.0001, 0.0005, 0.00005):
        workers.append(Process(target=process, args=({'learn_rate': i, 'gamma': 0.75},)))
    return workers

def gamma_train_test():
    workers = []
    for i in (0.99, 0.95, 0.85, 0.8, 0.75, 0.7):
        workers.append(Process(target=process, args=({'gamma': i},)))
    return workers

def worker_train_test():
    workers = []
    for i in (10, 100, 500, 1000, 5000, 10000):
        workers.append(Process(target=process, args=({'worker_number': i, 'gamma': 0.9, 'learn_rate': 0.00005},)))
    return workers

def sensor_train_test():
    workers = []
    for i in (500, 1000, 10000):
        workers.append(Process(target=process, args=({'sensor_number': i, 'gamma': 0.9, 'learn_rate': 0.00005},)))
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

def pho_test():
    workers = []
    for seed in range(400):
        for pho in [0.3, 0.5, 0.7, 0.9]:
            workers.append(Process(target=process, args=({'learn_rate': 0.00005, 'gamma': 0.9,
                          'worker_number': 5000, 'train': False, 'malicious': 0.5,
                          'console_log': False, 'seed': seed, 'pho': pho,
                          'suffix': '_' + str(seed) + '_pho_' + str(pho)},)))
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

    workers.extend(sensor_repeat_train())
    workers.extend(worker_repeat_train())
    # workers.extend(gamma_train_test())
    # workers.extend(worker_performance())
    for worker in workers:
        worker.daemon = True

    number = 40
    if len(workers) >= number:
        cnt = 0
        while cnt < int(len(workers) / number):
            for idx in range(cnt * number, min(len(workers), (cnt + 1) * number)):
                workers[idx].start()

            for idx in range(cnt * number, min(len(workers), (cnt + 1) * number)):
                workers[idx].join()
            cnt += 1
    else:
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

