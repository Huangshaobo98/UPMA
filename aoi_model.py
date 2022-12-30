from worker_model import Worker
from typing import List
from random import random as rand


class AoI:
    def __init__(self):
        self.__last_dummy = 0
        self.__last_report = 0

    def episode_clear(self):
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
            if rand() < worker.honest:
                self.__last_report = current_slot
                success_list.append(worker)
            else:
                fail_list.append(worker)
        self.dummy_report(workers, current_slot)
        return success_list, fail_list

    def dummy_report(self, workers: List[Worker], current_slot):
        fail_rate = 1.0
        for worker in workers:
            fail_rate *= (1 - worker.trust)
        self.__last_dummy = (1 - fail_rate) * (current_slot - self.__last_dummy) + self.__last_dummy
