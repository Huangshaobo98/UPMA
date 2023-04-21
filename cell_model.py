import time

import numpy as np
import os
from sensor_model import Sensor
from numpy import random
from math import sqrt, ceil
from data.data_clean import DataCleaner
import matplotlib.pyplot as plt
import time
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

    def max_cost_assignment_at_this_slot(self) -> float:
        ret = 0.0
        for worker in self.__workers_at_this_slot:
            ret += (worker.total_reward * ceil(worker.vitality * worker.trust))
        return ret

    def workload(self) -> int:
        ret = 0
        for worker in self.__workers_at_this_slot:
            ret += worker.vitality
        return ret

    def task_assignment_by_random(self,
                                  sensors_ref: list,
                                  workers_ref: list,
                                  current_slot: int,
                                  # reduce_rate: float,
                                  # random_flag: bool,
                                  cost: float,
                                  # workload_coefficient: float
                                  ):
        worker_random_idx = np.random.randint(0, len(workers_ref), int(cost * 5) + 1)
        sensor_random_idx = np.random.randint(0, len(sensors_ref), int(cost * 5) + 1)
        task_assignment_start = time.perf_counter()
        sensors = sensors_ref.copy()
        workers = workers_ref.copy()
        left_cost = cost
        if len(sensors) == 0 or len(workers) == 0:
            return 0, 0, 0, 0, 0, 0
        normal_task_number = 0
        malicious_task_number = 0
        normal_assignment = 0
        malicious_assignment = 0

        cnt = 0
        while left_cost > 0:
            worker = workers[worker_random_idx[cnt]]
            sensor = sensors[sensor_random_idx[cnt]]
            # if sensor.had_worker(worker):
            #     continue
            if left_cost - worker.total_reward < 0:
                break
            left_cost -= worker.total_reward
            sensor.add_worker(worker)
            if worker.malicious:
                malicious_assignment += worker.total_reward
                malicious_task_number += 1
            else:
                normal_assignment += worker.total_reward
                normal_task_number += 1
            cnt += 1
        task_assignment_end = time.perf_counter()
        task_assignment_time = task_assignment_end - task_assignment_start
        # print(task_assignment_time)
        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment, cost - left_cost, task_assignment_time


    def task_assignment_by_greed(self,
                                 sensors_ref: list,
                                 workers_ref: list,
                                 current_slot: int,
                                 # reduce_rate: float,
                                 # random_flag: bool, #废弃接口
                                 cost: float,
                                 # workload_coefficient: float
                                 ):
        # 要求sensor和worker为浅拷贝
        task_assignment_start = time.perf_counter()
        left_cost = cost
        sensors = sensors_ref.copy()
        sensor_len = len(sensors)
        workers = workers_ref.copy()
        workers.sort(key=lambda worker: worker.trust / worker.bid, reverse=True)
        worker_len = len(workers)
        if sensor_len == 0 or worker_len == 0:
            return 0, 0, 0, 0, 0, 0
        fail_exception = [1.0 for _ in range(sensor_len)]
        aoi = [sensor.get_observation_aoi(current_slot) for sensor in sensors]
        # sum_aoi = sum(aoi)

        workload = [ceil(worker.vitality * worker.trust) for worker in workers]
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
        while (left_cost >= workers[assignable_worker_idx].total_reward) and len(heap) > 0 and assignable_worker_idx < worker_len:
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
            left_cost -= workers[worker_idx].total_reward
            if workers[worker_idx].malicious:
                malicious_assignment += workers[worker_idx].total_reward
                malicious_task_number += 1
            else:
                normal_assignment += workers[worker_idx].total_reward
                normal_task_number += 1
        task_assignment_end = time.perf_counter()
        task_assignment_time = task_assignment_end - task_assignment_start
        # print(task_assignment_time)
        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment, cost - left_cost, task_assignment_time


    def task_assignment_by_genetic(self,
                                   sensors_ref: list,
                                   workers_ref: list,
                                   current_slot: int,
                                   cost: float,
                                   population_size: int = 40,
                                   mutation_n1_rate: float = 0.2,
                                   mutation_rate: float = 0.1,
                                   crossover_rate: float = 1,
                                   end_fix_round:int = 50,
                                   punishment_factor:float = 1000,
                                   tournament_number: int = 8,
                                 ):
        sensors = sensors_ref.copy()
        workers = workers_ref.copy()

        task_assignment_start = time.perf_counter()
        chromosome_length = 0

        normal_task_number = 0
        malicious_task_number = 0
        normal_assignment = 0
        malicious_assignment = 0

        worker_length = len(workers)
        sensor_length = len(sensors)
        if sensor_length == 0 or worker_length == 0:
            return 0, 0, 0, 0, 0, 0
        vitality_vector = np.zeros(shape=(worker_length, ), dtype=np.int32)
        for idx, worker in enumerate(workers):
            vitality_vector[idx] = ceil(worker.vitality * worker.trust)
            chromosome_length += vitality_vector[idx]

        wkr_index_vector = np.zeros(shape=(chromosome_length,), dtype=np.int32)
        bid_vector = np.zeros(shape=(chromosome_length,), dtype=np.float64)
        aoi_vector = np.zeros(shape=(sensor_length, ), dtype=np.float64)

        cnt = 0
        for idx, vitality in enumerate(vitality_vector):
            for _ in range(vitality):
                bid_vector[cnt] = workers[idx].total_reward
                wkr_index_vector[cnt] = idx
                cnt += 1

        for i in range(sensor_length):
            aoi_vector[i] = sensors[i].get_observation_aoi(current_slot)

        def mutation(chromosome):
            if np.random.rand() < mutation_n1_rate:
                mutation_point = np.random.randint(0, chromosome_length)
                chromosome[mutation_point] = -1
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(0, chromosome_length)
                chromosome[mutation_point] = np.random.randint(-1, sensor_length)
            return chromosome


        def crossover(chromosome_1, chromosome_2):
            new_chromosome_1 = chromosome_1.copy()
            new_chromosome_2 = chromosome_2.copy()
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(0, chromosome_length, (2,))
                min_point = np.min(crossover_point)
                max_point = np.max(crossover_point)
                new_chromosome_2[min_point:max_point+1] = chromosome_1[min_point:max_point+1]
                new_chromosome_1[min_point:max_point+1] = chromosome_2[min_point:max_point+1]

            return new_chromosome_1, new_chromosome_2

        def fitness_calculate(chromosome):
            last_wkr = -1
            this_wkr_worked_sensor = set()
            fail_rate = np.ones(shape=(len(sensors, )), dtype=np.float64)
            for idx, sen_idx in enumerate(chromosome):
                if wkr_index_vector[idx] != last_wkr:
                    last_wkr = wkr_index_vector[idx]
                    this_wkr_worked_sensor.clear()
                if sen_idx in this_wkr_worked_sensor:
                    continue
                fail_rate[sen_idx] *= (1 - workers[wkr_index_vector[idx]].trust)
                this_wkr_worked_sensor.add(sen_idx)

            left_cost = cost - np.sum(bid_vector[chromosome >= 0])
            punishment = (left_cost * punishment_factor) if left_cost < 0 else 0
            fitness = np.sum((1 - fail_rate) * aoi_vector) - punishment

            return fitness

        def selection(fitness_list):
            new_population_idx = np.zeros(shape=(population_size,), dtype=np.int32)
            for idx in range(population_size):
                tournament_idx = np.random.choice([i for i in range(2 * population_size)], tournament_number, replace=False)
                winner_idx = np.argmax(fitness_list[tournament_idx])
                new_population_idx[idx] = tournament_idx[winner_idx]
            return new_population_idx

        population = np.random.randint(0, len(sensors), (population_size * 2, chromosome_length), dtype=np.int32) # 初始种群
        population[np.random.rand(population_size * 2, chromosome_length) < 0.5] = -1
        max_fitness = -99999999.9
        max_fitness_chromosome = None

        max_fitness_stopped = 0

        fitness_array = np.zeros(shape=(population_size * 2, ), dtype=np.float64)
        for i in range(population_size):
            fitness_array[i] = fitness_calculate(population[i])
            if fitness_array[i] > max_fitness:
                max_fitness = fitness_array[i]
                max_fitness_chromosome = population[i]

        while max_fitness_stopped < end_fix_round:
            # new_population = np.zeros(shape=(population_size * 2, chromosome_length), dtype=np.int32)
            # new_population[:population_size, :] = population
            for idx in range(int(population_size / 2)):
                new_chromosome_1, new_chromosome_2 = crossover(population[idx * 2], population[idx * 2 + 1])
                population[population_size + idx * 2] = mutation(new_chromosome_1)
                population[population_size + idx * 2 + 1] = mutation(new_chromosome_2)
                if not np.all(population[population_size + idx * 2] == population[idx * 2]):
                    fitness_array[population_size + idx * 2] = fitness_calculate(population[population_size + idx * 2])
                else:
                    fitness_array[population_size + idx * 2] = fitness_array[idx * 2]
                if fitness_array[population_size + idx * 2] > max_fitness:
                    max_fitness_stopped = 0
                    max_fitness = fitness_array[population_size + idx * 2]
                    max_fitness_chromosome = population[population_size + idx * 2]

                if not np.all(population[population_size + idx * 2 + 1] == population[idx * 2 + 1]):
                    fitness_array[population_size + idx * 2 + 1] = fitness_calculate(population[population_size + idx * 2 + 1])
                else:
                    fitness_array[population_size + idx * 2 + 1] = fitness_array[idx * 2 + 1]
                if fitness_array[population_size + idx * 2 + 1] > max_fitness:
                    max_fitness_stopped = 0
                    max_fitness = fitness_array[population_size + idx * 2 + 1]
                    max_fitness_chromosome = population[population_size + idx * 2 + 1]

            next_population_idx = selection(fitness_array)
            population[:population_size] = population[next_population_idx]
            fitness_array[:population_size] = fitness_array[next_population_idx]

            max_fitness_stopped += 1

        total_cost = 0
        kmn = set()
        for idx, sensor_idx in enumerate(max_fitness_chromosome):
            if sensor_idx == -1 or (sensor_idx, wkr_index_vector[idx]) in kmn:
                continue
            worker_ptr = workers[wkr_index_vector[idx]]
            sensor_ptr = sensors[sensor_idx]
            sensor_ptr.add_worker(worker_ptr)
            total_cost += worker_ptr.total_reward
            if worker_ptr.malicious:
                malicious_assignment += worker_ptr.total_reward
                malicious_task_number += 1
            else:
                normal_assignment += worker_ptr.total_reward
                normal_task_number += 1
            if total_cost >= cost:
                break
        task_assignment_end = time.perf_counter()
        task_assignment_time = task_assignment_end - task_assignment_start
        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment, total_cost, task_assignment_time


    # 废弃方法，不使用
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

    def clear_workers(self):
        self.__workers_at_this_slot.clear()

    def task_assignment_(self, current_slot, assignment_method='greedy', cost=0):
        # if assignment_method == 'sort':
        #     malicious_number, normal_number, malicious_assignment, normal_assignment, cost = self.task_assignment_by_sort(self.sensors, self.workers,
        #                                                                            current_slot, False)
        if assignment_method == 'random':
            malicious_number, normal_number, malicious_assignment, normal_assignment, cost, task_assignment_time \
                = self.task_assignment_by_random(sensors_ref=self.sensors,
                                                 workers_ref=self.workers,
                                                 current_slot=current_slot,
                                                 cost=cost)
        elif assignment_method == 'greedy':
            malicious_number, normal_number, malicious_assignment, normal_assignment, cost, task_assignment_time \
                = self.task_assignment_by_greed(sensors_ref=self.sensors,
                                                workers_ref=self.workers,
                                                current_slot=current_slot,
                                                cost=cost)
        elif assignment_method == 'genetic':
            malicious_number, normal_number, malicious_assignment, normal_assignment, cost, task_assignment_time \
                = self.task_assignment_by_genetic(sensors_ref=self.sensors,
                                                  workers_ref=self.workers,
                                                  current_slot=current_slot,
                                                  cost=cost)
        else:
            assert False
        self.__workers_at_this_slot.clear()
        return malicious_number, normal_number, malicious_assignment, normal_assignment, cost, task_assignment_time

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

    def plot_cell(self, axis, linewidth=1):
        x_val = [item + self.__x_location
                 for item in [self.__length * sqrt(3) / 2, 0, -self.__length * sqrt(3) / 2,
                              -self.__length * sqrt(3) / 2, 0, self.__length * sqrt(3) / 2,
                              self.__length * sqrt(3) / 2]]
        y_val = [item + self.__y_location
                 for item in [self.__length / 2, self.__length, self.__length / 2, -self.__length / 2,
                              -self.__length, -self.__length / 2, self.__length / 2]]
        return axis.plot(x_val, y_val, color='k', linewidth=linewidth)

    @staticmethod
    def plot_cells(cleaner: DataCleaner, cells: np.ndarray):
        name = 'x' + str(cleaner.x_limit) + '_y' + str(cleaner.y_limit) + ('_uniform' if not cleaner.Norm else '_norm') \
               + ('_no_dataset' if cleaner.No_data_set_need else '_with_dataset')
        [sensor_x, sensor_y] = Sensor.get_all_locations()
        map_fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = map_fig.add_subplot(111)
        np.random.seed(10)
        if not cleaner.No_data_set_need:
            sp = cleaner.worker_coordinate().shape[0]
            sample_nodes = cleaner.worker_coordinate()[np.random.choice(sp, int(sp/50), False), :]
            work_p = ax.scatter(cleaner.worker_coordinate()[:, 0], cleaner.worker_coordinate()[:, 1], color='gray', marker='o', s=0.01, alpha=0.5)
        for row in cells:
            for cell in row:
                cell.plot_cell(ax)

        sen_p = ax.scatter(sensor_x, sensor_y, color='royalblue', marker='o', s=15)
        uav_p = ax.scatter(cleaner.cell_coordinate[5, 5][0], cleaner.cell_coordinate[5, 5][1], color='r', marker='o', s=25)
        plt.xlim(cleaner.x_range[0], cleaner.x_range[1])
        plt.ylim(cleaner.y_range[0], cleaner.y_range[1])

        if cleaner.Range_is_angle:
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
        else:
            plt.xlabel("x(meter)")
            plt.ylabel("y(meter)")
        plt.legend((sen_p, uav_p), ("SNs", "UAV"), loc='lower right')
        if not os.path.exists('./figure/'):
            os.mkdir('./figure/')
        plt.savefig("./figure/{}.tif".format(name))
        plt.savefig("./figure/{}.jpg".format(name))
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
    # cleaner = DataCleaner(x_limit=8,
    #                       y_limit=8,
    #                       x_range=[0, 2000],  # 可以是角度，也可以是距离
    #                       y_range=[0, 1800],
    #                       range_is_angle=False, # 很重要的参数，确定是角度的话需要乘以地球系数
    #                       Norm=False,
    #                       Norm_centers=[],
    #                       Norm_centers_ratio=[], # 每一个分布中心所占比率
    #                       Norm_sigma=[],       # 正态分布的方差
    #                       Norm_gain=[],        # 正态分布系数，用于控制器辐射半径大小
    #                       No_data_set_need=True)
    # cleaner = DataCleaner(
    #     x_limit=6,
    #     y_limit=6,
    #     x_range=[0, 2000],  # 可以是角度，也可以是距离
    #     y_range=[0, 1800],
    #     range_is_angle=False,  # 很重要的参数，确定是角度的话需要乘以地球系数
    #     Norm=True,
    #     Norm_centers=[[500, 600], [1400, 1100]],
    #     Norm_centers_ratio=[0.4, 0.6],  # 每一个分布中心所占比率
    #     Norm_sigma=[1, 1],  # 正态分布的方差
    #     Norm_gain=[400, 600],  # 正态分布系数，用于控制器辐射半径大小
    #     No_data_set_need=True)
    cleaner = DataCleaner()
    c = Cell.uniform_generator_with_position(cleaner, 5000)
    Cell.plot_cells(cleaner, c)
