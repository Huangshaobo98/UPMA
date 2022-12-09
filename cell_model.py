import numpy as np

from sensor_model import Sensor
from numpy import random
from math import sqrt
from data.data_clean import DataCleaner
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, x, y, position, side_length):
        self.__x = x
        self.__y = y
        self.__x_location = position[0]
        self.__y_location = position[1]
        self.__sensors = []
        self.__length = side_length

        self.__workers_at_this_slot = []

        # self.__map_style = g.map_style

        # self.__x_location = 1.5 * (self.__x - self.__y) * self.__length
        # self.__y_location = sqrt(3) / 2 * (self.__x + self.__y) * self.__length
        # if g.map_style == 'h':
        #     self.__x_location = 1.5 * (self.__x - self.__y) * self.__length
        #     self.__y_location = sqrt(3) / 2 * (self.__x + self.__y) * self.__length
        # else:
        #     self.__x_location = self.__length * self.__x
        #     self.__y_location = self.__length * self.__y

    @property
    def index(self):
        return [self.__x, self.__y]

    def add_worker(self, worker):
        self.__workers_at_this_slot.append(worker)

    def task_assignment(self, current_slot):
        # 任务分配
        sensors = self.__sensors
        sorted_sensor = sorted(self.__sensors,
                               key=lambda sensor: sensor.get_observation_aoi(current_slot),
                               reverse=True)
        sorted_time = [sensor.get_observation_aoi(current_slot) for sensor in sorted_sensor]
        sensor_cnt = len(sorted_sensor)
        if sensor_cnt == 0:
            return

        trust_index = np.zeros(shape=(sensor_cnt,), dtype=float)

        worker = self.__workers_at_this_slot
        worker_cnt = len(self.__workers_at_this_slot)



        sorted_workers = sorted(self.__workers_at_this_slot,
                                key=lambda worker: worker.trust,
                                reverse=True)

        worker_trusts = [worker.trust for worker in sorted_workers]
        if worker_cnt > 1:
            pass
        worker_vitality = [worker.vitality for worker in sorted_workers]

        sensor_ptr = 0
        satisfied_cnt = 0
        worker_ptr = 0
        while satisfied_cnt < sensor_cnt and worker_ptr < worker_cnt:
            if worker_vitality[worker_ptr] == 0:
                worker_ptr += 1
                continue
            if trust_index[sensor_ptr] > 1.0:
                sensor_ptr = (sensor_ptr + 1) % sensor_cnt
                continue
            sorted_sensor[sensor_ptr].add_worker(sorted_workers[worker_ptr])
            worker_vitality[worker_ptr] -= 1
            trust_index[sensor_ptr] += sorted_workers[worker_ptr].trust
            if trust_index[sensor_ptr] >= 1.0:
                satisfied_cnt += 1
            sensor_ptr = (sensor_ptr + 1) % sensor_cnt

        self.__workers_at_this_slot.clear()

    def episode_clear(self):
        for sensor in self.__sensors:
            sensor.episode_clear()

    def add_sensor(self, sensor):
        self.__sensors.append(sensor)

    @property
    def position(self):
        return [self.__x_location, self.__y_location]

    @property
    def sensors(self):
        return self.__sensors

    @property
    def sensor_number(self):
        return len(self.__sensors)

    def plot_cell(self, axis):
        x_val = [item + self.__x_location
                 for item in [self.__length * sqrt(3) / 2, 0, -self.__length * sqrt(3) / 2,
                              -self.__length * sqrt(3) / 2, 0, self.__length * sqrt(3) / 2,
                              self.__length * sqrt(3) / 2]]
        y_val = [item + self.__y_location
                 for item in [self.__length / 2, self.__length, self.__length / 2, -self.__length / 2,
                              -self.__length, -self.__length / 2, self.__length / 2]]
        axis.plot(x_val, y_val, color='k', linewidth=1)

    @staticmethod
    def plot_cells(cells: np.ndarray):
        [sensor_x, sensor_y] = Sensor.get_all_locations()
        map_fig = plt.figure(figsize=(10, 8), dpi=450)
        ax = map_fig.add_subplot(111)
        for row in cells:
            for cell in row:
                cell.plot_cell(ax)
        ax.scatter(sensor_x, sensor_y, color='r')
        plt.savefig('./cell_sensors.jpg')

    def uav_visited(self, current_slot):
        for sensor in self.__sensors:
            sensor.report_by_uav(current_slot)

    def worker_visited(self, current_slot):
        # 刷新状态，用于车辆访问的情况
        for sensor in self.__sensors:
            sensor.report_by_workers(current_slot)

    def get_observation_aoi(self, current_slot):
        ret = 0.0
        for sensor in self.__sensors:
            ret += sensor.get_observation_aoi(current_slot)
        return ret

    def get_real_aoi(self, current_slot):
        ret = 0.0
        for sensor in self.__sensors:
            ret += sensor.get_real_aoi(current_slot)
        return ret

    @staticmethod
    def uniform_generator_with_position(x_limit: int,
                                        y_limit: int,
                                        positions: np.ndarray,
                                        sensor_number: int,
                                        side_length: float,
                                        seed: object = 10) -> np.ndarray:

        ret_cell = np.empty(shape=(x_limit, y_limit), dtype=object)
        for x in range(x_limit):
            for y in range(y_limit):
                ret_cell[x, y] = Cell(x, y, positions[x][y], side_length)

        random.seed(seed)
        sensor_x = random.randint(0, x_limit, sensor_number)
        sensor_y = random.randint(0, y_limit, sensor_number)

        sensor_x_diff = random.uniform(-side_length * sqrt(3) / 2, side_length * sqrt(3) / 2, sensor_number)
        sensor_y_diff = np.array([random.uniform(abs(x_diff) / sqrt(3) - side_length,
                                                 - abs(x_diff) / sqrt(3) + side_length) for x_diff in sensor_x_diff])

        # print("sensors_x" + str(sensor_x[:20]) + 'sensor_y' + str(sensor_y[:20]))
        # print("x_diff" + str(sensor_x_diff[:20]) + 'y_diff' + str(sensor_y_diff[:20]))
        for i in range(sensor_number):
            Sensor(i, sensor_x_diff[i], sensor_y_diff[i], ret_cell[sensor_x[i], sensor_y[i]])

        return ret_cell



if __name__ == '__main__':
    cleaner = DataCleaner()
    c = Cell.uniform_generator_with_position(cleaner.x_limit,
                                             cleaner.y_limit,
                                             cleaner.cell_coordinate,
                                             5000,
                                             cleaner.side_length)
    Cell.plot_cells(c)
