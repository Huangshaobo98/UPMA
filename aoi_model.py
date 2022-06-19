from worker_model import Worker
from typing import List
from random import random as rand


class AoI:
    def __init__(self):
        self.__last_dummy = 0
        self.__last_report = 0

    def get_real_aoi(self, cur_slot):
        return cur_slot - self.__last_report

    def get_observation_aoi(self, cur_slot):
        return cur_slot - self.__last_dummy

    def report(self, workers: List[Worker], current_slot):
        for worker in workers:
            if rand() < worker.get_honest():
                self.__last_report = current_slot
                break
        self.dummy_report(workers, current_slot)

    def dummy_report(self, workers: List[Worker], current_slot):
        max_trust = 0
        for worker in workers:
            max_trust = max(max_trust, worker.get_trust())
        self.__last_dummy = max_trust * (current_slot - self.__last_dummy) - self.__last_dummy
