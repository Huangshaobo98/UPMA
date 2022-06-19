from aoi_model import AoI
from worker_model import Worker
from typing import List

class Sensor:
    __location_x = []
    __location_y = []

    def __init__(self, x_location: float, y_location: float, cell_x: int, cell_y: int):
        self.__x = x_location
        self.__y = y_location
        self.__cell_x = cell_x
        self.__cell_y = cell_y
        Sensor.__location_x.append(self.__x)
        Sensor.__location_y.append(self.__y)

        self.__aoi = AoI()

    def get_observation_aoi(self, cur_slot):
        return self.__aoi.get_observation_aoi(cur_slot)

    def get_real_aoi(self, cur_slot):
        return self.__aoi.get_real_aoi(cur_slot)

    def get_cell_index(self):
        return self.__cell_x, self.__cell_x

    def get_position(self):
        return self.__x, self.__y

    def report_by_uav(self, current_slot):
        self.__aoi.report_by_uav(current_slot)

    def report_by_workers(self, workers: List[Worker], current_slot):
        self.__aoi.report_by_worker(workers, current_slot)

    @staticmethod
    def get_all_locations():
        return Sensor.__location_x, Sensor.__location_y
