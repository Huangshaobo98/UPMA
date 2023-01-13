import numpy as np
import io
from sensor_model import Sensor
from numpy import random
from math import sqrt
from data.data_clean import DataCleaner
import matplotlib.pyplot as plt
from PIL import Image
import heapq
class Cell:
    def __init__(self, x, y, position, side_length):
        self.__x = x
        self.__y = y
        self.__x_location = position[0]
        self.__y_location = position[1]
        self.__sensors = []
        self.__length = side_length
        self.__energy_consume = 0
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

    def task_assignment_by_greed(self, sensors_ref: list, workers_ref: list, current_slot: int, reduce_rate: float, random_flag: bool):
        # 要求sensor和worker为浅拷贝

        sensors = sensors_ref.copy()
        sensor_len = len(sensors)

        workers = workers_ref.copy()
        if not random_flag:
            workers.sort(key=lambda worker: worker.trust, reverse=True)
        worker_len = len(workers)

        fail_exception = [1.0 for _ in range(sensor_len)]
        aoi = [sensor.get_observation_aoi(current_slot) for sensor in sensors]
        sum_aoi = sum(aoi)

        workload = [worker.vitality for worker in workers]
        heap = []
        kmn = set()
        reduce_aoi = 0
        assignable_worker_idx = 0

        normal_task_number = 0
        malicious_task_number = 0
        normal_assignment = 0
        malicious_assignment = 0

        for idx, sensor in enumerate(sensors):
            heapq.heappush(heap, (- aoi[idx] * fail_exception[idx], idx))
        while len(heap) > 0 and assignable_worker_idx < worker_len and reduce_aoi / sum_aoi <= reduce_rate:
            if workload[assignable_worker_idx] == 0:
                assignable_worker_idx += 1
                continue
            (neg_reduce_aoi, sensor_idx) = heapq.heappop(heap)   # 取出需要被急切优化的node
            worker_idx = assignable_worker_idx
            while worker_idx < worker_len and ((worker_idx, sensor_idx) in kmn) and workload[worker_idx] > 0:
                worker_idx += 1     # 当前worker已经被分配了采集sensor_index的任务，继续找下一个worker
            if worker_idx == worker_len:
                continue            # 找不到可以分配任务的worker，放弃当前sensor的任务分配，找下一个合适的sensor
            sensors[sensor_idx].add_worker(workers[worker_idx]) #找到了，则分配
            kmn.add((worker_idx, sensor_idx))
            workload[worker_idx] -= 1
            old_reduce_aoi = aoi[sensor_idx] * (1 - fail_exception[sensor_idx])
            fail_exception[sensor_idx] *= (1 - workers[worker_idx].trust)
            new_neg_reduce_aoi = aoi[sensor_idx] * (1 - fail_exception[sensor_idx])
            heapq.heappush(heap, (new_neg_reduce_aoi, sensor_idx))
            benefit = new_neg_reduce_aoi - old_reduce_aoi
            reduce_aoi = reduce_aoi + benefit
            if workers[worker_idx].malicious:
                malicious_assignment += benefit
                malicious_task_number += 1
            else:
                normal_assignment += benefit
                normal_task_number += 1
        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment

    def task_assignment_by_sort(self, sensors_ref: list, workers_ref: list, current_slot: int, random_assignment: bool):

        sorted_sensors =  sensors_ref.copy()
        sorted_workers = workers_ref.copy()

        if random_assignment:
            sorted_sensors.sort(key=lambda sensor: sensor.get_observation_aoi(current_slot), reverse=True)
            sorted_workers.sort(key=lambda worker: worker.trust, reverse=True)

        sensor_cnt = len(sorted_sensors)
        if sensor_cnt == 0:
            return 0, 0

        trust_index = np.zeros(shape=(sensor_cnt,), dtype=float)
        # worker = self.__workers_at_this_slot
        worker_cnt = len(sorted_workers)

        # worker_trusts = [worker.trust for worker in sorted_workers]
        worker_vitality = [worker.vitality for worker in sorted_workers]

        normal_task_number = 0
        malicious_task_number = 0

        malicious_assignment = 0
        normal_assignment = 0

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
            sorted_sensors[sensor_ptr].add_worker(sorted_workers[worker_ptr])
            worker_vitality[worker_ptr] -= 1
            trust_index[sensor_ptr] += sorted_workers[worker_ptr].trust

            benefit = sorted_workers[worker_ptr].get_observation_aoi(current_slot)
            if sorted_workers[worker_ptr].malicious:
                malicious_assignment += benefit
            else:
                normal_assignment += benefit

            if trust_index[sensor_ptr] >= 1.0:
                satisfied_cnt += 1
            sensor_ptr = (sensor_ptr + 1) % sensor_cnt

        # self.__workers_at_this_slot.clear()

        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment

    def task_assignment_(self, current_slot, assignment_method='greedy', convergence_factor=0.875):
        if assignment_method == 'sort':
            malicious_number, normal_number, malicious_assignment, normal_assignment = self.task_assignment_by_sort(self.sensors, self.workers,
                                                                                   current_slot, False)
        elif assignment_method == 'random':
            malicious_number, normal_number, malicious_assignment, normal_assignment = self.task_assignment_by_greed(self.sensors, self.workers,
                                                                                    current_slot, convergence_factor, True)
        elif assignment_method == 'greedy':
            malicious_number, normal_number, malicious_assignment, normal_assignment = self.task_assignment_by_greed(self.sensors, self.workers,
                                                                                    current_slot, convergence_factor, False)
        else:
            assert False
        self.__workers_at_this_slot.clear()
        return malicious_number, normal_number, malicious_assignment, normal_assignment

    def task_assignment(self, current_slot, random_assignment: bool = False):
        # 任务分配
        # sensors = self.__sensors
        sorted_sensor =  self.__sensors if random_assignment \
            else sorted(self.__sensors,key=lambda sensor: sensor.get_observation_aoi(current_slot), reverse=True)
        # sorted_time = [sensor.get_observation_aoi(current_slot) for sensor in sorted_sensor]
        sensor_cnt = len(sorted_sensor)
        if sensor_cnt == 0:
            return 0, 0

        trust_index = np.zeros(shape=(sensor_cnt,), dtype=float)

        # worker = self.__workers_at_this_slot
        worker_cnt = len(self.__workers_at_this_slot)

        sorted_workers = self.__workers_at_this_slot if random_assignment \
            else sorted(self.__workers_at_this_slot, key=lambda worker: worker.trust, reverse=True)

        # worker_trusts = [worker.trust for worker in sorted_workers]
        worker_vitality = [worker.vitality for worker in sorted_workers]

        malicious_assignment = 0
        normal_assignment = 0

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
            if sorted_workers[worker_ptr].malicious:
                malicious_assignment += 1
            else:
                normal_assignment += 1
            if trust_index[sensor_ptr] >= 1.0:
                satisfied_cnt += 1
            sensor_ptr = (sensor_ptr + 1) % sensor_cnt

        self.__workers_at_this_slot.clear()

        return malicious_assignment, normal_assignment

    def episode_clear(self):
        for sensor in self.__sensors:
            sensor.episode_clear()

    def add_sensor(self, sensor):
        self.__sensors.append(sensor)

    def add_collection_consume(self, sensor_collect_consume):
        self.__energy_consume += sensor_collect_consume

    @property
    def position(self):
        return [self.__x_location, self.__y_location]

    @property
    def sensors(self):
        return self.__sensors

    @property
    def workers(self):
        return self.__workers_at_this_slot

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
    def plot_cells(cleaner: DataCleaner, cells: np.ndarray):
        [sensor_x, sensor_y] = Sensor.get_all_locations()
        map_fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = map_fig.add_subplot(111)
        np.random.seed(10)
        sp = cleaner.worker_coordinate().shape[0]
        sample_nodes = cleaner.worker_coordinate()[np.random.choice(sp, int(sp/50), False), :]
        work_p = ax.scatter(cleaner.worker_coordinate()[:, 0], cleaner.worker_coordinate()[:, 1], color='gray', marker='o', s=0.01, alpha=0.5)
        for row in cells:
            for cell in row:
                cell.plot_cell(ax)

        sen_p = ax.scatter(sensor_x, sensor_y, color='royalblue', marker='o', s=2)
        uav_p = ax.scatter(cleaner.cell_coordinate[5, 5][0], cleaner.cell_coordinate[5,5][1], color='r', marker='o', s=25)
        plt.xlim(cleaner.x_range[0], cleaner.x_range[1])
        plt.ylim(cleaner.y_range[0], cleaner.y_range[1])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend((sen_p, uav_p), ("SNs", "UAV"), loc='lower right')
        plt.savefig("/figure/cell_sensors.tif")
        plt.savefig("/figure/cell_sensors.jpg")
        print('ok')


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

    @property
    def collection_consume(self)->float:
        return self.__energy_consume

    @staticmethod
    def uniform_generator_with_position(cleaner,
                                        sensor_number: int,
                                        seed: object = 10) -> np.ndarray:

        ret_cell = np.empty(shape=(cleaner.x_limit, cleaner.y_limit), dtype=object)
        cell_positions = cleaner.cell_coordinate
        side_length = cleaner.side_length
        sensor_cell = cleaner.sensor_cell
        sensor_diff = cleaner.sensor_diff

        for x in range(cleaner.x_limit):
            for y in range(cleaner.y_limit):
                ret_cell[x, y] = Cell(x, y, cell_positions[x][y], side_length)

        # print("sensors_x" + str(sensor_x[:20]) + 'sensor_y' + str(sensor_y[:20]))
        # print("x_diff" + str(sensor_x_diff[:20]) + 'y_diff' + str(sensor_y_diff[:20]))
        for i in range(sensor_number):
            ret_cell[sensor_cell[i, 0], sensor_cell[i, 1]].add_collection_consume(cleaner.energy_consume(sensor_diff[i,0], sensor_diff[i,1]))
            Sensor(i, sensor_diff[i,0], sensor_diff[i,1], ret_cell[sensor_cell[i,0], sensor_cell[i,1]])
        return ret_cell



if __name__ == '__main__':
    cleaner = DataCleaner()
    c = Cell.uniform_generator_with_position(cleaner, 5000)
    Cell.plot_cells(cleaner, c)
