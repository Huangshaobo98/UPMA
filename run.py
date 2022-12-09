import os

from main import Main
from multiprocessing import Process


def process(kwargs):
    print("Setting " + str(kwargs) + " begin on process id: " + str(os.getpid()) + "\n", end="")
    processor = Main(kwargs)
    processor.start()
    processor.end()

def learn_train_test():
    workers = []
    for i in (0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
        workers.append(Process(target=process, args=({'learn_rate': i},)))

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

def gamma_train_test():
    workers = []
    for i in (0.99, 0.95, 0.85, 0.8, 0.75, 0.7):
        workers.append(Process(target=process, args=({'gamma': i},)))

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    gamma_train_test()

