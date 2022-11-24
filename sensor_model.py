from aoi_model import AoI
from worker_model import Worker
from collections import defaultdict


class Sensor:
    __location_x = []
    __location_y = []
    __num_index = 0

    def __init__(self, x_location: float, y_location: float, cell_x: int, cell_y: int):
        self.__id = Sensor.__num_index
        Sensor.__num_index += 1
        self.__x = x_location
        self.__y = y_location
        self.__cell_x = cell_x
        self.__cell_y = cell_y
        Sensor.__location_x.append(self.__x)
        Sensor.__location_y.append(self.__y)
        self.__worker_report_list = []
        self.__aoi = AoI()
        self.__worker_success_dict = defaultdict(int)
        self.__worker_fail_dict = defaultdict(int)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    @property
    def index(self):
        return self.__id

    def get_observation_aoi(self, cur_slot):
        return self.__aoi.get_observation_aoi(cur_slot)

    def get_real_aoi(self, cur_slot):
        return self.__aoi.get_real_aoi(cur_slot)

    def get_cell_index(self):
        return self.__cell_x, self.__cell_x

    def get_position(self):
        return self.__x, self.__y

    def report_by_uav(self, current_slot):
        self.__aoi.report_by_uav(current_slot)      # 对uav访问节点的aoi刷新
        for worker, value in self.__worker_success_dict.items():    # 对采集过此节点的worker进行汇报成功
            worker.add_success(value)
        for worker, value in self.__worker_fail_dict.items():
            worker.add_fail(value)

    def report_by_workers(self, current_slot):
        if len(self.__worker_report_list) == 0:
            return
        [success_list, fail_list] = self.__aoi.report_by_worker(self.__worker_report_list, current_slot)
        for worker in success_list:
            self.__worker_success_dict[worker] += 1
        for worker in fail_list:
            self.__worker_fail_dict[worker] += 1

    def add_worker(self, worker: Worker):
        self.__worker_report_list.append(worker)

    def clear(self):
        self.__worker_success_dict.clear()
        self.__worker_fail_dict.clear()
        self.__aoi.clear()

    @staticmethod
    def get_all_locations():
        return Sensor.__location_x, Sensor.__location_y
