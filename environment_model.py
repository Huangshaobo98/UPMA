# from sensor_model import Sensor
import random
# import sys

from cell_model import Cell
from worker_model import WorkerBase, UAV, Worker, MobilePolicy
from global_parameter import Global as g
from agent_model import DQNAgent, State
import numpy as np
from persistent_model import Persistent
from logger import Logger
from typing import List
from data.data_clean import DataCleaner
from energy_model import Energy

class Compare:
    def __init__(self,
                 x_limit,
                 y_limit,
                 method):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.method = method
        self.actions = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 0], [0, 0]]
        self.cell_visited = np.zeros(shape=(self.x_limit, self.y_limit), dtype=bool)

    def run(self, prev_state: State):
        [cx, cy] = prev_state.position
        energy_left = prev_state.energy
        energy_consume = Energy.move_energy_cost()
        obv_aoi = prev_state.observation_aoi_state
        if self.method == "RR":
            return self.RoundRobin(cx, cy, energy_left, energy_consume)
        elif self.method == "Greedy":
            return self.Greedy(cx, cy, obv_aoi, energy_left, energy_consume)
        elif self.method == "CCPP":
            return self.CCPP(cx, cy, obv_aoi, energy_left, energy_consume)
        else:
            assert False

    def RoundRobin(self, cx, cy, energy_left, energy_consume):
        if energy_left - energy_consume < 0:
            return 6

        if cx == 0:
            if cy == 0:
                return 2 # 起始状态移动
            else:
                return 0 # 返回初始位置

        if cy % 2 == 0:
            if cx == self.x_limit - 1:
                return 4
            else:
                return 2
        else:
            if cx == 1:
                if cy == self.y_limit - 1:
                    return 5
                return 4
            else:
                return 5

    def Greedy(self, cx, cy, obv_aoi, energy_left, energy_consume):
        if energy_left - energy_consume < 0:
            return 6
        # if random.random() < 0.05:
        #     return random.randint(0, 5)
        max_aoi = 0
        index = -1
        for idx, act in enumerate(self.actions):
            [nx, ny] = [cx + act[0], cy + act[1]]
            if nx >= 0 and nx < self.x_limit and ny >= 0 and ny < self.y_limit:
                if obv_aoi[nx, ny] > max_aoi:
                    max_aoi = obv_aoi[nx, ny]
                    index = idx

        return index

    def get_path(self, cx, cy, obv_aoi):
        actions = np.array([(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 0)])
        cur_pos = [cx, cy]
        path = np.empty(shape=(self.x_limit, self.y_limit), dtype=object)
        path[tuple(cur_pos)] = np.array([], dtype=int)  #之前的代码是直接使用位置来写的，这里我觉得可以改成使用动作
        path_AoI = np.zeros(shape=(self.x_limit, self.y_limit))
        cover_state = np.zeros(shape=(self.x_limit, self.y_limit))
        cover_state[tuple(cur_pos)] = 1
        layer_cells = np.array([cur_pos])
        layers = np.empty(shape=(2 * max(self.x_limit, self.y_limit) - 2,), dtype=object)
        count = 0
        while not cover_state.all():
            new_layer_cells = np.empty(shape=(0, 2), dtype=np.int32)
            for cell in layer_cells:
                for act_idx, action in enumerate(actions):
                    temp_pos = action + cell
                    if 0 <= temp_pos[0] < self.x_limit and 0 <= temp_pos[1] < self.y_limit \
                            and cover_state[tuple(temp_pos)] == 0:
                        if path_AoI[tuple(temp_pos)] < path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1:
                            path_AoI[tuple(temp_pos)] = path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1
                            path[tuple(temp_pos)] = np.append(path[tuple(cell)], act_idx)
                            new_layer_cells = np.vstack((new_layer_cells, temp_pos))
            layer_cells = np.unique(new_layer_cells, axis=0)
            for cell in layer_cells:
                cover_state[tuple(cell)] = 1
            layers[count] = layer_cells
            count += 1
        return path, path_AoI, layers

    def CCPP(self, cx, cy, obv_aoi, energy_left, energy_consume):
        if energy_left - energy_consume < 0:
            return 6
        path, path_AoI, layers = self.get_path(cx, cy, obv_aoi)
        self.cell_visited[cx, cy] = True
        for i in range(self.x_limit):
            for j in range(self.y_limit):
                if obv_aoi[i, j] <= 0.00001:
                    self.cell_visited[i, j] = True
        if self.cell_visited.all():
            self.cell_visited = np.zeros(shape=(self.x_limit, self.y_limit), dtype=bool)
            self.cell_visited[cx, cy] = True
        max_val = 0
        max_path_act = np.empty(shape=(0, 2), dtype=np.int32)
        for layer in layers:
            for cell in layer:
                if self.cell_visited[tuple(cell)] == 0:
                    if path_AoI[tuple(cell)] > max_val:
                        max_val = path_AoI[tuple(cell)]
                        max_path_act = path[tuple(cell)][0]
            if max_val > 0:
                break
        return max_path_act



class Environment:
    def __init__(self,
                 train: bool,
                 continue_train: bool,
                 compare: bool,
                 compare_method: str,
                 sensor_number: int,
                 worker_number: int,
                 max_episode: int,
                 # max_slot: int,
                 batch_size: int,
                 epsilon_decay: float,
                 learn_rate: float,
                 gamma: float,
                 detail: bool,
                 seed: int,
                 mali_rate: float,
                 win_len: int,
                 pho: float,
                 random_task_assignment: bool,
                 assignment_reduce_rate: float,
                 cleaner: DataCleaner
                 ):

        # 一些比较固定的参数
        # self.__charge_cells = g.charge_cells
        # self.__uav_fix = g.uav_start_fix
        # self.__uav_start_location = g.uav_start_location
        # self.__initial_trust = g.initial_trust

        self.__train = train  # 训练模式
        self.__compare = compare
        self.__compare_method = compare_method
        # self.__continue_train = continue_train  # 续训模式

        self.random_task_assignment = random_task_assignment
        # self.__sensor_number = sensor_number
        # self.__worker_number = worker_number

        self.train_by_real = False
        self.cleaner = cleaner
        self.compare_act = Compare(cleaner.x_limit, cleaner.y_limit, compare_method)
        self.__detail = detail

        self.mali_rate = mali_rate
        self.reduce_rate = assignment_reduce_rate
        self.sensor_number = sensor_number
        # 训练相关
        self.__max_episode = max_episode if train else g.default_test_episode
        self.__episode = 0
        self.__max_slot = cleaner.slot_number
        self.__current_slot = 0
        self.__sum_slot = 0

        # reward相关
        self.__no_power_punish = cleaner.x_limit * cleaner.y_limit
        self.__hover_punish = cleaner.x_limit * cleaner.y_limit
        self.__batch_size = batch_size

        # episode data for persistent
        self.__episode_real_aoi = []
        self.__episode_observation_aoi = []
        self.__episode_energy = []
        self.__episode_reward = []


        if not self.__train:
            self.__slot_real_aoi = np.zeros(shape=(cleaner.slot_number, cleaner.x_limit, cleaner.y_limit), dtype=np.float64)
            self.__slot_obv_aoi = np.zeros(shape=(cleaner.slot_number, cleaner.x_limit, cleaner.y_limit), dtype=np.float64)
            self.__episode_data = {
                'real_aoi_by_slot': np.zeros(shape=(self.__max_episode, cleaner.x_limit, cleaner.y_limit), dtype=np.float64),
                'obv_aoi_by_slot': np.zeros(shape=(self.__max_episode, cleaner.x_limit, cleaner.y_limit), dtype=np.float64),
                'visit_time': np.zeros(shape=(self.__max_episode, cleaner.x_limit, cleaner.y_limit), dtype=np.int16),
                'avg_real_aoi': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'avg_obv_aoi': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'norm': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'bad_trust': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'good_trust': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'bad_task_number': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'good_task_number': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'bad_assignment': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'good_assignment': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'reward': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'energy': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'actual_slot': np.zeros(shape=(self.__max_episode,), dtype=np.int16),
            }
        # energy
        Energy.init(cleaner)

        # agent
        if not compare:
            self.__agent = DQNAgent(cleaner.cell_limit, action_size=7, gamma=gamma, epsilon=Persistent.trained_epsilon(),
                                    epsilon_decay=epsilon_decay, lr=learn_rate, train=train, continue_train=continue_train,
                                    model_path=Persistent.model_path())
        else:
            self.__agent = None

        # network model
        self.__cell = Cell.uniform_generator_with_position(cleaner,
                                                           sensor_number)

        WorkerBase.set_cleaner(cleaner)
        self.__uav = UAV(cleaner.cell_limit)
        np.random.seed(seed)
        malicious = np.random.random(size=(10357,)) < mali_rate

        self.__worker = [Worker(i, cleaner.worker_position[i],
                                malicious=malicious[i], direct_window=win_len, recom_window=win_len*2,
                                pho=pho)
                         for i in range(worker_number)]

        cleaner.free_memory()
        # sensor_x, sensor_y = Sensor.get_all_locations()
        # Persistent.save_network_model(g.cell_length, self.__cell_limit, np.stack([sensor_x, sensor_y]))

    def get_cell_observation_aoi(self, current_slot):
        ret = np.empty((self.cleaner.x_limit, self.cleaner.y_limit), dtype=np.float64)
        for x in range(self.cleaner.x_limit):
            for y in range(self.cleaner.y_limit):
                ret[x, y] = self.__cell[x, y].get_observation_aoi(current_slot)
        return ret

    def get_cell_real_aoi(self, current_slot):
        ret = np.empty((self.cleaner.x_limit, self.cleaner.y_limit), dtype=np.float64)
        for x in range(self.cleaner.x_limit):
            for y in range(self.cleaner.y_limit):
                ret[x, y] = self.__cell[x, y].get_real_aoi(current_slot)
        return ret

    # def get_analyze(self):
    #     cell_aoi = sum(sum(self.get_cell_real_aoi(self.__current_slot)))
    #     uav_location = self.position_state
    #     return cell_aoi, uav_location

    def get_energy_state(self) -> float:
        ret = self.__uav.energy
        return ret

    def get_position_state(self) -> List[int]:
        dx_dy = [dx, dy] = self.__uav.position
        assert len(dx_dy) == 2 and 0 <= dx < self.cleaner.x_limit and 0 <= dy < self.cleaner.y_limit
        return dx_dy

    def get_charge_state(self):
        return self.__uav.charge_state

    def get_network_state(self, slot: int) -> State:
        # 获取网络观测aoi/实际aoi/无人机位置/能量信息
        observation_aoi = self.get_cell_observation_aoi(slot)  # 观测aoi
        real_aoi = self.get_cell_real_aoi(slot)  # 实际aoi
        position_state = self.get_position_state()  # uav位置
        energy_state = self.get_energy_state()  # 能量
        charge_state = self.get_charge_state()
        return State(real_aoi, observation_aoi, position_state, energy_state, charge_state)

    def get_current_network_state(self) -> State:
        # 获取当前(“当前”指的是无人机移动前)网络状态信息
        return self.get_network_state(self.__current_slot)

    def get_next_network_state(self) -> State:
        # 获取无人机移动后网络状态信息
        return self.get_network_state(self.__current_slot + 1)

    def cell_update_by_uav(self):
        [ux, uy] = self.get_position_state()   # 获取无人机的位置信息
        self.__cell[ux, uy].uav_visited(self.__current_slot + 1)  # note: 无人机需要在下个时隙才能到达目标

    def reward_calculate(self, prev_state: State, next_state: State, hover: bool, charge: bool):
        # 因为这里是训练得到的reward，因此用real_aoi进行计算
        # 这里虽然无人机可能会花费几个slot来换电池，但是我们对于模型的预测仍然采用下一个时隙的结果进行预测
        reward = - np.sum(next_state.real_aoi_state if self.train_by_real else next_state.observation_aoi_state) \
                 + self.__no_power_punish * g.energy_reward_calculate(next_state.energy_state[0]) \
                 - self.__hover_punish * (hover and not charge)

        return reward

    def uav_step(self):
        # uav步进
        Logger.log("\r\n" + "-" * 36 + " UAV step. " + "-" * 36)
        prev_state = self.get_current_network_state()

        # if not self.__train:
        #     self.__slot_real_aoi[self.__current_slot-1] = prev_state.real_aoi_state
        # 根据上述状态，利用神经网络寻找最佳动作
        if not self.__compare:
            uav_action_index, action_values = self.__agent.act(prev_state, self.train_by_real)
        else:
            uav_action_index = self.compare_act.run(prev_state)
            action_values = []
        # 将index转换为二维方向dx_dy
        uav_action = MobilePolicy.get_action(uav_action_index)
        # 无人机移动，主要是动作改变和充电的工作
        self.__uav.act(uav_action, self.__cell)
        # 无人机执行所在小区的数据收集/评估worker的信任等工作
        self.cell_update_by_uav()

        next_state = self.get_next_network_state()

        # hover = True if charge_state or ((prev_position == next_position).all()) else False   # 悬浮状态需要计算

        # if charge_state:  这里原意是如果进入了充电，将耗费几个slot，新的模型只需要耗掉1个slot即可
        #     self.__charge_slot = self.__slot_for_charge
        #     hover = False

        # next_observation_aoi = self.get_cell_observation_aoi(self.__current_slot)
        # next_real_aoi = self.get_cell_real_aoi(self.__current_slot)
        # next_energy = self.energy_state()

        return prev_state, uav_action_index, action_values, next_state

    def cell_step(self, cell_pos_to_refresh: set):
        [malicious_number, normal_number, malicious_aoi, normal_aoi] = [0, 0, 0, 0]
        for x, y in cell_pos_to_refresh:
            if [x, y] == self.get_position_state():
                continue
            # if self.__train:
            #     malicious_assignment, normal_assignment = self.__cell[x, y].task_assignment(self.__current_slot, self.random_task_assignment)    # 任务分配
            # else:
            method = 'greedy' if not self.random_task_assignment else 'random'
            malicious_task_number, normal_task_number, malicious_assignment, normal_assignment = self.__cell[x, y].task_assignment_(self.__current_slot, method, self.reduce_rate)  #
            malicious_number += malicious_task_number
            normal_number += normal_task_number
            malicious_aoi += malicious_assignment
            normal_aoi += normal_assignment
            self.__cell[x, y].worker_visited(self.__current_slot)     # 任务执行

        return malicious_number, normal_number, malicious_aoi, normal_aoi


    def workers_step(self):
        # 确定小区内存在那些车辆，并交由这些小区自行处理(进行任务分配和执行)
        cell_pos_to_refresh = set()
        for worker in self.__worker:
            next_position = worker.move(self.__current_slot)
            if next_position is None:
                continue
            [x, y] = next_position
            self.__cell[x, y].add_worker(worker)
            cell_pos_to_refresh.add((x, y))

        return cell_pos_to_refresh

        # Logger.log("\r\n" + "-" * 34 + " Workers step. " + "-" * 34)
        # cell_pos_to_refresh = set()
        # for worker in self.__worker:
        #     next_position = worker.move(self.__current_slot)
        #     if next_position is None:
        #         continue
        #     [x, y] = next_position
        #     if worker.work(self.__cell[x, y]):
        #         cell_pos_to_refresh.add((x, y))
        # if len(cell_pos_to_refresh) == 0:
        #     Logger.log("No workers are working.")
        # for tup in cell_pos_to_refresh:
        #     self.__cell[tup[0], tup[1]].worker_visited(self.__current_slot)

    def worker_trust_refresh(self):

        for work in self.__worker:
            [trust, direct, recom] = work.update_trust()
            # Logger.log("Worker {}: trust {:.4f}, direct {:.4f}, recom {:.4f}"
            #            .format(work.index, trust, direct, recom))

    def uav_step_state_detail(self, prev_state: State, next_state: State,
                              action: List[int], action_values: List[float], reward: float, epsilon: float
                              ):
        act_msg = "1. Action details: \r\n"
        if len(action_values) == 0:
            act_msg += "Random action, selected action: {}.\r\n\r\n".format(action)
        else:
            act_msg += "Action values: {}, selected action: {}.\r\n\r\n".format(action_values, action)

        uav_msg = "2. Agent(UAV) state details: \r\n" \
                  + "Position state: {} -> {}, charge state: {} -> {}, " \
                    "energy state: {:.4f} -> {:.4f}, reward: {:.4f}, " \
                    "random action rate: {:.6f}.\r\n\r\n" \
                  .format(prev_state.position, next_state.position,
                          prev_state.charge, next_state.charge,
                          prev_state.energy, next_state.energy,
                          reward, epsilon)

        env_msg = "3. Age of Information(AoI) state: \r\n" \
                  + "Average real aoi state: {:.4f} -> {:.4f}, average observation state: {:.4f} -> {:.4f}\r\n" \
                  .format(prev_state.average_real_aoi_state, next_state.average_real_aoi_state,
                          prev_state.average_observation_aoi_state, next_state.average_observation_aoi_state)

        if self.__detail:   # 这里显示日志
            real_aoi_msg = ""
            observation_aoi_msg = ""

            prev_real_aoi_list = str(prev_state.real_aoi_state).split('\n')
            next_real_aoi_list = str(next_state.real_aoi_state).split('\n')
            prev_observation_aoi_list = str(prev_state.observation_aoi_state).split('\n')
            next_observation_aoi_list = str(next_state.observation_aoi_state).split('\n')

            for prev_real_aoi, next_real_aoi, \
                prev_observation_aoi, next_observation_aoi\
                    in zip(prev_real_aoi_list, next_real_aoi_list,
                           prev_observation_aoi_list, next_observation_aoi_list):
                real_aoi_msg += (str(prev_real_aoi) + "\t\t|\t" + str(next_real_aoi) + "\r\n")
                observation_aoi_msg += (str(prev_observation_aoi) + "\t\t|\t" + str(next_observation_aoi) + "\r\n")

            real_aoi_space_number = max(real_aoi_msg.find('\t') - len("Prev real AoI state: "), 0)
            observation_aoi_space_num = max(observation_aoi_msg.find('\t') - len("Prev observation AoI state: "), 0)
            real_aoi_msg = "Prev real AoI state: " + " " * real_aoi_space_number \
                           + "\t\t|\tNext real AoI state: \n" + real_aoi_msg
            observation_aoi_msg = "Prev observation AoI state: " + " " * observation_aoi_space_num \
                                  + "\t\t|\tNext observation AoI state: \n" + observation_aoi_msg

            env_msg += (real_aoi_msg + "\r\n" + observation_aoi_msg)
        return act_msg + uav_msg + env_msg

    def slot_step(self):
        # 整合test和train的slot步进方法
        refresh_cells = self.workers_step()  # worker先行移动
        [malicious_number, normal_number,
         malicious_assignment, normal_assignment] = self.cell_step(refresh_cells)        # 小区进行任务分配和执行

        done = False

        # prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, \
        # prev_position, next_position, prev_energy, next_energy, \
        # prev_charge_state, next_charge_state, uav_action_index = self.uav_step()  # 根据训练/测试方式，选择按观测aoi/实际aoi作为输入

        prev_state, action, action_values, next_state = self.uav_step()
        # Logger.log("State before UAV action: " + str(prev_state))
        # Logger.log("State after UAV action: " + str(next_state))
        if self.__current_slot >= self.__max_slot or next_state.energy <= 0.0:  # 这里需要考虑电量问题?电量不足时是否直接结束状态
            done = True

        hover = True if (prev_state.position == next_state.position
                         and next_state.position not in g.charge_cells) else False
        charge = self.get_charge_state()
        reward = self.reward_calculate(prev_state, next_state, hover, charge)

        Logger.log(self.uav_step_state_detail(prev_state, next_state, action, action_values,
                                                  reward, self.__agent.epsilon if self.__train else 0))


        if self.__train:
            self.__agent.memorize(prev_state, action, next_state, reward, done)

        self.__episode_real_aoi.append(np.sum(next_state.real_aoi))
        self.__episode_observation_aoi.append(np.sum(next_state.observation_aoi))
        self.__episode_energy.append(next_state.energy)
        self.__episode_reward.append(reward)

        self.worker_trust_refresh()  # worker信任刷新

        if not self.__train:
            good_trust = []
            bad_trust = []
            for worker in self.__worker:
                if worker.malicious:
                    bad_trust.append(worker.trust)
                else:
                    good_trust.append(worker.trust)
            # persist_data = {
            #     'episode': self.__episode,
            #     'slot': self.__current_slot,
            #     'sum real aoi': np.sum(next_state.real_aoi),
            #     'average real aoi': np.average(next_state.real_aoi),
            #     'sum observation aoi': np.sum(next_state.observation_aoi),
            #     'average observation aoi': np.average(next_state.observation_aoi),
            #     'hover': hover,
            #     'charge': self.get_charge_state(),
            #     'uav position x': next_state.position[0],
            #     'uav position y': next_state.position[1],
            #     'reward': reward,
            #     'energy': next_state.energy,
            #     'energy left rate': next_state.energy_state.sum(),
            #     'norm': np.linalg.norm(x=prev_state.observation_aoi_state-prev_state.real_aoi_state, ord=2),
            #     'good trust': np.average(good_trust),
            #     'bad trust': np.average(bad_trust),
            #     'malicious assignment': malicious_assignment,
            #     'normal assignment': normal_assignment,
            #     'epsilon': self.__agent.epsilon if self.__train else 0,
            # }
            self.__slot_real_aoi[self.__current_slot - 1] = prev_state.real_aoi_state
            self.__slot_obv_aoi[self.__current_slot - 1] = prev_state.observation_aoi_state
            self.__episode_data['visit_time'][self.__episode - 1, next_state.position[0], next_state.position[1]] += 1
            self.__episode_data['avg_real_aoi'][self.__episode - 1, self.__current_slot - 1] = \
                np.sum(prev_state.real_aoi) / self.sensor_number
            self.__episode_data['avg_obv_aoi'][self.__episode - 1, self.__current_slot - 1] = \
                np.sum(prev_state.observation_aoi) / self.sensor_number
            self.__episode_data['norm'][self.__episode - 1, self.__current_slot - 1] = \
                np.linalg.norm(x=prev_state.observation_aoi_state - prev_state.real_aoi_state, ord=2)
            self.__episode_data['bad_trust'][self.__episode - 1, self.__current_slot - 1] = np.average(bad_trust)
            self.__episode_data['good_trust'][self.__episode - 1, self.__current_slot - 1] = np.average(good_trust)
            self.__episode_data['bad_assignment'][self.__episode - 1, self.__current_slot - 1] = malicious_assignment
            self.__episode_data['good_assignment'][self.__episode - 1, self.__current_slot - 1] = normal_assignment
            self.__episode_data['bad_task_number'][self.__episode - 1, self.__current_slot - 1] = malicious_number
            self.__episode_data['good_task_number'][self.__episode - 1, self.__current_slot - 1] = normal_number
            self.__episode_data['reward'][self.__episode - 1, self.__current_slot - 1] = reward
            self.__episode_data['energy'][self.__episode - 1, self.__current_slot - 1] = prev_state.energy
            self.__episode_data['actual_slot'][self.__episode - 1] = self.__current_slot
            # Persistent.save_data(persist_data)

        if self.__train and self.__sum_slot % 100 == 0:
            self.__agent.update_target_model()

        if self.__train and len(self.__agent.memory) >= self.__batch_size:
            self.__agent.replay(self.__batch_size)

        return done

    def save_train_episode_data_to_npz(self):
        if not self.__train:
            save_path = Persistent.npz_path()
            np.savez(save_path,
                     real_aoi_by_slot=self.__episode_data['real_aoi_by_slot'],
                     obv_aoi_by_slot=self.__episode_data['obv_aoi_by_slot'],
                     visit_time=self.__episode_data['visit_time'],
                     avg_real_aoi=self.__episode_data['avg_real_aoi'],
                     avg_obv_aoi=self.__episode_data['avg_obv_aoi'],
                     norm=self.__episode_data['norm'],
                     bad_trust=self.__episode_data['bad_trust'],
                     good_trust=self.__episode_data['good_trust'],
                     bad_assignment=self.__episode_data['bad_assignment'],
                     good_assignment=self.__episode_data['good_assignment'],
                     bad_task_number=self.__episode_data['bad_task_number'],
                     good_task_number=self.__episode_data['good_task_number'],
                     reward=self.__episode_data['reward'],
                     energy=self.__episode_data['energy'],
                     actual_slot=self.__episode_data['actual_slot'])

    def episode_clear(self):
        # 刷新状态，重新开始
        for row_cells in self.__cell:
            for cell in row_cells:
                cell.episode_clear()
        np.random.seed(self.__episode)
        malicious = np.random.random(size=(10357,)) < self.mali_rate
        for idx, work in enumerate(self.__worker):
            work.episode_clear(malicious[idx])
        self.__uav.clear()
        self.__episode_real_aoi.clear()
        self.__episode_observation_aoi.clear()
        self.__episode_energy.clear()
        self.__episode_reward.clear()
        self.__slot_real_aoi = np.zeros(shape=(self.cleaner.slot_number, self.cleaner.x_limit, self.cleaner.y_limit), dtype=np.float64)
        self.__slot_obv_aoi = np.zeros(shape=(self.cleaner.slot_number, self.cleaner.x_limit, self.cleaner.y_limit), dtype=np.float64)

    def episode_step(self):
        begin_slot = self.__sum_slot
        for slot in range(1, self.__max_slot + 1):
            self.__current_slot = slot
            self.__sum_slot += 1
            Logger.log(("=" * 84 + "\r\n" + "{:^84}\r\n" + "=" * 84)
                       .format("Episode: {}, slot: {}, sum slot: {}."
                               .format(self.__episode, self.__current_slot, self.__sum_slot)))
            if self.slot_step():
                break
        episode_data = {
            'episode': self.__episode,
            'average real aoi': np.average(self.__episode_real_aoi),
            'average observation aoi': np.average(self.__episode_observation_aoi),
            'average energy': np.average(self.__episode_energy),
            'average reward': np.average(self.__episode_reward),
            'begin slot': begin_slot,
            'slot number': self.__current_slot,
            'epsilon': self.__agent.epsilon if self.__train else 0,
        }
        Persistent.save_episode_data(episode_data)
        if not self.__train:
            self.__episode_data['real_aoi_by_slot'][self.__episode - 1] = np.average(self.__slot_real_aoi, axis=0)
            self.__episode_data['obv_aoi_by_slot'][self.__episode - 1] = np.average(self.__slot_obv_aoi, axis=0)
        self.episode_clear()
        if self.__train:
            self.__agent.save(Persistent.model_path())
            if self.__episode % 25 == 0:
                self.__agent.save(Persistent.model_directory() + '/backup_{}.h5'.format(self.__episode))

    def start(self):

        np.set_printoptions(suppress=True, precision=3)
        for episode in range(Persistent.trained_episode() + 1, Persistent.trained_episode() + self.__max_episode + 1):
            self.__episode = episode
            self.episode_step()
        self.save_train_episode_data_to_npz()

        # 用于绘制热图的data
        # if not self.__train:
        #     file_name = Persistent.data_directory() + "/slot_aoi"
        #     if self.__compare:
        #         file_name += ('_' + self.__compare_method)
        #     file_name += '.npy'
        #     np.save(file_name, np.average(self.__slot_real_aoi, axis=0))

