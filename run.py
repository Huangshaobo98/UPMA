import os
import signal
from main import Main
from multiprocessing import Process

workers = []

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

if __name__ == '__main__':

    signal.signal(signal.SIGTERM, signal_handler)
    print("master process: " + str(os.getpid()) + "\n", end="")

    workers.extend(sensor_train_test())
    # workers.extend(gamma_train_test())

    for worker in workers:
        worker.daemon = True

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

