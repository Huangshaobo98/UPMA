from main import Main
from multiprocessing import Process


def process(kwargs):
    processor = Main(kwargs)
    processor.start()
    processor.end()


if __name__ == '__main__':
    workers = []
    for i in (0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005):
        workers.append(Process(target=process, args=({'learn_rate': i},)))

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


