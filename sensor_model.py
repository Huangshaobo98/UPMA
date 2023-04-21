from aoi_model import AoI
from worker_model import Worker
from collections import defaultdict
# from cell_model import Cell


class Sensor:
    __location_x = []
    __location_y = []
    __num_index = 0

    def __init__(self, index, x_diff: float, y_diff: float, cell):
        self.__id = index

        [cell_x_position, cell_y_position] = cell.position
        self.__x_position = x_diff + cell_x_position
        self.__y_position = y_diff + cell_y_position

        Sensor.__location_x.append(self.__x_position)
        Sensor.__location_y.append(self.__y_position)

        [cell_x, cell_y] = cell.index
        self.__cell_x = cell_x
        self.__cell_y = cell_y

        self.__worker_report_list = []
        self.__aoi = AoI()
        self.__worker_success_dict = defaultdict(int)
        self.__worker_fail_dict = defaultdict(int)

        cell.add_sensor(self)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def __lt__(self, other):
        return self.index < other.index

    @property
    def index(self):
        return self.__id

    def get_observation_aoi(self, cur_slot):
        return self.__aoi.get_observation_aoi(cur_slot)

    def get_real_aoi(self, cur_slot):
        return self.__aoi.get_real_aoi(cur_slot)

    @property
    def cell_index(self):
        return [self.__cell_x, self.__cell_y]

    @property
    def position(self):
        return [self.__x_position, self.__y_position]

    def report_by_uav(self, current_slot):
        self.__aoi.report_by_uav(current_slot)      # 对uav访问节点的aoi刷新
        for worker, value in self.__worker_success_dict.items():    # 对采集过此节点的worker进行汇报成功
            worker.add_direct_success(value)
        self.__worker_success_dict.clear()
        for worker, value in self.__worker_fail_dict.items():
            worker.add_direct_fail(value)
        self.__worker_fail_dict.clear()

    def report_by_workers(self, current_slot):
        if len(self.__worker_report_list) == 0:
            return
        [success_list, fail_list] = self.__aoi.report_by_worker(self.__worker_report_list, current_slot)
        self.__worker_report_list.clear()
        for idx, worker in enumerate(success_list):
            for n in range(idx + 1, len(success_list)):
                worker.add_recom_result(success_list[n], True)
            self.__worker_success_dict[worker] += 1

        for idx, worker in enumerate(fail_list):
            for n in range(idx + 1, len(fail_list)):
                worker.add_recom_result(fail_list[n], False)
            self.__worker_fail_dict[worker] += 1

    def add_worker(self, worker: Worker):
        self.__worker_report_list.append(worker)

    def had_worker(self, worker: Worker) -> bool:
        for wkr in self.__worker_report_list:
            if id(wkr) == id(worker):
                return True
        return False

    def episode_clear(self):
        self.__worker_report_list.clear()
        self.__worker_success_dict.clear()
        self.__worker_fail_dict.clear()
        self.__aoi.episode_clear()

    @staticmethod
    def get_all_locations():
        return Sensor.__location_x, Sensor.__location_y
