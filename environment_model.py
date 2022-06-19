from cell_model import uniform_generator
from worker_model import UAV, Worker
from random import randint
from global_ import Global
from math import sqrt

class Environment:
    def __init__(self):
        g = Global()
        x_limit = g["x_limit"]
        y_limit = g["y_limit"]
        worker_number = g["worker_number"]
        self.__uav = UAV(randint(0, x_limit), randint(0, y_limit))
        self.__cell = uniform_generator()
        self.__worker = [Worker(randint(0, x_limit), randint(0, y_limit)) for _ in range(worker_number)]
        self.__sec_per_slot = g["cell_length"] * sqrt(3) / g["uav_speed"] if g["map_style"] == 'h' else g["cell_length"] / g["uav_speed"]

    def get_cell_observation_aoi(self, current_slot):
        ret = []
        for Rows in self.__cell:
            ret_row = []
            for cell in Rows:
                ret_row.append(cell.get_observation_aoi(current_slot))
            ret.append(ret_row)
        return ret

    def get_cell_real_aoi(self, current_slot):
        ret = []
        for Rows in self.__cell:
            ret_row = []
            for cell in Rows:
                ret_row.append(cell.get_real_aoi(current_slot))
            ret.append(ret_row)
        return ret

    def uav_step(self, uav_action, current_slot):
        # 返回: 观测aoi状态，实际aoi状态，当前无人机所在小区
        prev_observation_aoi = self.get_cell_observation_aoi(current_slot)
        prev_real_aoi = self.get_cell_real_aoi(current_slot)
        [prev_location, next_location, prev_energy, next_energy, slot_cost] = self.__uav.action(uav_action) # 对无人机的状态进行更新
        # 这里虽然无人机可能会花费几个slot来换电池，但是我们对于模型的预测仍然采用下一个时隙的结果进行预测
        self.__cell[next_location].uav_visited(current_slot)

        next_observation_aoi = self.get_cell_observation_aoi(current_slot)
        next_real_aoi = self.get_cell_real_aoi(current_slot)

        return prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, prev_location, next_location, prev_energy, next_energy, slot_cost

    def workers_step(self, current_slot):
        pass




