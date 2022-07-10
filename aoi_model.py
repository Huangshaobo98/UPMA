from worker_model import Worker
from typing import List
from random import random as rand


class AoI:
    def __init__(self):
        self.__last_dummy = 0
        self.__last_report = 0

    def clear(self):
        self.__last_dummy = 0
        self.__last_report = 0

    def get_real_aoi(self, cur_slot):
        return cur_slot - self.__last_report

    def get_observation_aoi(self, cur_slot):
        return cur_slot - self.__last_dummy

    def report_by_uav(self, current_slot):
        self.__last_dummy = current_slot
        self.__last_report = current_slot

    def report_by_worker(self, workers: List[Worker], current_slot):
        success_list = []
        fail_list = []
        for worker in workers:
            if rand() < worker.get_honest():
                self.__last_report = current_slot
                success_list.append(worker)
            else:
                fail_list.append(worker)
        self.dummy_report(workers, current_slot)
        return success_list, fail_list

    def dummy_report(self, workers: List[Worker], current_slot):
        max_trust = 0
        for worker in workers:
            max_trust = max(max_trust, worker.get_trust())
        self.__last_dummy = max_trust * (current_slot - self.__last_dummy) + self.__last_dummy
