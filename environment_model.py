# from sensor_model import Sensor
import heapq
import time
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
from compare import Compare
from cost_helper import Cost_helper
from math import ceil

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
                 task_assignment_policy: str,
                 no_uav: bool,
                 # assignment_reduce_rate: float,
                 cost_limit: int,
                 max_energy: float,
                 basic_reward_for_worker: float,
                 max_bid_for_worker: float,
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

        self.no_uav = no_uav
        # self.__continue_train = continue_train  # 续训模式

        self.cost_limit = (sensor_number if cost_limit == -1 else cost_limit)
        self.workload_cost_coefficient = 1.0

        self.task_assignment_policy = task_assignment_policy
        # self.__sensor_number = sensor_number
        # self.__worker_number = worker_number

        self.train_by_real = False
        self.cleaner = cleaner
        self.compare_act = Compare(cleaner.x_limit, cleaner.y_limit, compare_method, max_energy)
        self.__detail = detail

        self.mali_rate = mali_rate
        # self.reduce_rate = assignment_reduce_rate
        self.sensor_number = sensor_number
        # 训练相关
        self.__max_episode = max_episode if train else (g.default_test_episode if not task_assignment_policy == 'genetic' else 5)
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
                'move_action': np.empty(shape=(self.__max_episode,), dtype=dict),
                'uav_move_decision_time': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
                'task_assignment_time': np.zeros(shape=(self.__max_episode, self.__max_slot), dtype=np.float64),
            }
            for n in range(self.__max_episode):
                self.__episode_data['move_action'][n] = {}
        # energy
        Energy.init(cleaner)

        # agent
        if not compare and not self.no_uav:
            self.__agent = DQNAgent(cleaner.cell_limit, action_size=7, gamma=gamma, epsilon=Persistent.trained_epsilon(),
                                    epsilon_decay=epsilon_decay, lr=learn_rate, train=train, continue_train=continue_train,
                                    model_path=Persistent.model_path())
        else:
            self.__agent = None

        # network model
        self.__cell = Cell.uniform_generator_with_position(cleaner,
                                                           sensor_number)

        WorkerBase.set_cleaner(cleaner)
        self.__uav = UAV(cleaner.cell_limit, max_energy)
        np.random.seed(seed)
        malicious = np.random.random(size=(10357,)) < mali_rate

        self.__worker = [Worker(i, cleaner.worker_position[i],
                                malicious=malicious[i], direct_window=win_len, recom_window=win_len*2,
                                pho=pho, basic_reward=basic_reward_for_worker, max_bid=max_bid_for_worker)
                         for i in range(worker_number)]

        cleaner.free_memory(worker_number)
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

        if self.no_uav:
            return prev_state, -1, [], prev_state, 0
        # if not self.__train:
        #     self.__slot_real_aoi[self.__current_slot-1] = prev_state.real_aoi_state
        # 根据上述状态，利用神经网络寻找最佳动作

        uav_move_decision_start = time.perf_counter()

        if not self.__compare:
            uav_action_index, action_values = self.__agent.act(prev_state, self.train_by_real)
        else:
            uav_action_index = self.compare_act.run(prev_state, Energy.move_energy_cost())
            action_values = []
        # 将index转换为二维方向dx_dy
        uav_action = MobilePolicy.get_action(uav_action_index, prev_state.position[1])

        uav_move_decision_end = time.perf_counter()
        uav_move_decision_time = uav_move_decision_end - uav_move_decision_start

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

        return prev_state, uav_action_index, action_values, next_state, uav_move_decision_time

    def cost_assignment(self, cell_pos_to_refresh: set):
        wait_cost_assignment = []
        finished_assignment = []
        left_cost = self.cost_limit
        for x, y in cell_pos_to_refresh:
            wait_cost_assignment.append(Cost_helper(x=x, y=y,
                                                    aoi=self.__cell[x, y].get_observation_aoi(self.__current_slot),
                                                    max_cost=self.__cell[x, y].max_cost_assignment_at_this_slot()))

        while left_cost > 1.0 and len(finished_assignment) < len(cell_pos_to_refresh):
            # 在无剩余成本或已经全部分配完成时，可以结束分配
            assign_cost_this_round = 0
            total_wait_aoi = sum([cell.aoi for cell in wait_cost_assignment])              # 可分配小区aoi
            if total_wait_aoi == 0:
                break
            new_wait_cost_assignment = []
            for cell in wait_cost_assignment:
                this_cell_max_left_assign = cell.max_cost - cell.cost  # 剩余可分配成本
                this_cell_should_assign = left_cost * cell.aoi / total_wait_aoi                    # 实际应该分配成本
                this_cell_actual_assign = min(this_cell_max_left_assign, this_cell_should_assign)
                cell.cost += this_cell_actual_assign
                assign_cost_this_round += this_cell_actual_assign
                if abs(cell.cost - cell.max_cost) <= 1e-4:
                    finished_assignment.append(cell)
                else:
                    new_wait_cost_assignment.append(cell)
            left_cost -= assign_cost_this_round
            wait_cost_assignment = new_wait_cost_assignment
        finished_assignment += wait_cost_assignment
        return finished_assignment

    def global_assignment(self, cell_pos_to_refresh: set):
        assert self.task_assignment_policy == 'g-greedy'
        task_assignment_start = time.perf_counter()
        cost_data = np.zeros(shape=(self.cleaner.x_limit, self.cleaner.y_limit), dtype=np.float64)
        normal_task_number = 0
        malicious_task_number = 0
        normal_assignment = 0
        malicious_assignment = 0
        # 浅拷贝元素, 并对其进行排序组合
        worker_sensor_pairs = []
        sum_workload = 0
        for (x, y) in cell_pos_to_refresh:
            if len(self.__cell[x, y].sensors) == 0:
                continue
            worker_load_pair = []
            negaoi_sensor_pair = []
            for worker in self.__cell[x, y].workers:
                workload = ceil(worker.trust * worker.vitality)
                sum_workload += workload
                worker_load_pair.append([worker, workload])

            for sensor in self.__cell[x, y].sensors:
                negaoi_sensor_pair.append([-sensor.get_observation_aoi(self.__current_slot), sensor])

            worker_load_pair.sort(key=lambda worker: worker[0].trust, reverse=True)
            heapq.heapify(negaoi_sensor_pair)
            worker_sensor_pairs.append([worker_load_pair, negaoi_sensor_pair])

        left_cost = self.cost_limit
        actual_cost = 0
        while left_cost > 0 and sum_workload > 0:
            max_aoi_reduce = 0
            max_aoi_idx = -1
            remove_list = []
            ## 空耗计数器
            # __nop = 0
            ##
            for idx, [worker_load_pair, negaoi_sensor_pair] in enumerate(worker_sensor_pairs):  # 按小区找最大aoi减少量的组合
                if len(worker_load_pair) == 0:
                    remove_list.append(idx) # remove list还没实现....
                    continue
                if worker_load_pair[0][0].trust * (-negaoi_sensor_pair[0][0]) > max_aoi_reduce:
                    max_aoi_idx = idx
                    max_aoi_reduce = worker_load_pair[0][0].trust * (-negaoi_sensor_pair[0][0])
                ## 时间空耗，模拟未经优化的场景
                # for _ in worker_load_pair:
                #     __nop += 1
                # for _ in negaoi_sensor_pair:
                #     __nop += 1

            to_update_worker_load_pair, to_update_negaoi_sensor_pair = worker_sensor_pairs[max_aoi_idx]
            best_worker = to_update_worker_load_pair[0][0]
            to_update_worker_load_pair[0][1] -= 1
            update_trust = best_worker.trust

            if(to_update_worker_load_pair[0][1] == 0):
                to_update_worker_load_pair.pop(0)
            pop_negaoi, pop_sensor = heapq.heappop(to_update_negaoi_sensor_pair)
            heapq.heappush(to_update_negaoi_sensor_pair, [pop_negaoi * (1 - update_trust), pop_sensor])

            for item in remove_list[::-1]:
                worker_sensor_pairs.pop(item)

            sum_workload -= 1
            left_cost -= best_worker.total_reward

            if left_cost > 0:
                pop_sensor.add_worker(best_worker)
                [x, y] = pop_sensor.cell_index
                cost_data[x, y] += best_worker.total_reward
                if best_worker.malicious:
                    malicious_assignment += best_worker.total_reward
                    malicious_task_number += 1
                else:
                    normal_assignment += best_worker.total_reward
                    normal_task_number += 1
            else:
                break
        task_assignment_end = time.perf_counter()
        task_assignment_time = task_assignment_end - task_assignment_start
        for (x, y) in cell_pos_to_refresh:
            self.__cell[x, y].worker_visited(self.__current_slot)
            self.__cell[x, y].clear_workers()
        return malicious_task_number, normal_task_number, malicious_assignment, normal_assignment, cost_data, task_assignment_time

    def cell_assignment(self, cell_pos_to_refresh: set):
        assert self.task_assignment_policy == 'greedy' \
               or self.task_assignment_policy == 'genetic' \
               or self.task_assignment_policy == 'random'
        [malicious_number, normal_number, malicious_aoi, normal_aoi] = [0, 0, 0, 0]
        cell_cost_assignment = self.cost_assignment(cell_pos_to_refresh)
        cost_data = np.zeros(shape=(self.cleaner.x_limit, self.cleaner.y_limit))
        total_task_assignment_time = 0
        for cell_data in cell_cost_assignment:
            [x, y] = [cell_data.x, cell_data.y]
            malicious_task_number, normal_task_number, malicious_assignment, normal_assignment, cost, task_assignment_time \
                = self.__cell[x, y].task_assignment_(self.__current_slot, self.task_assignment_policy, cell_data.cost)  #
            total_task_assignment_time += task_assignment_time
            malicious_number += malicious_task_number
            normal_number += normal_task_number
            malicious_aoi += malicious_assignment
            normal_aoi += normal_assignment
            cost_data[x, y] = cost
            self.__cell[x, y].worker_visited(self.__current_slot)  # 任务执行
        return malicious_number, normal_number, malicious_aoi, normal_aoi, cost_data, total_task_assignment_time

    def cell_step(self, cell_pos_to_refresh: set):
        cell_pos_to_refresh.discard(tuple(self.get_position_state()))
        if self.task_assignment_policy == 'g-greedy':
            return self.global_assignment(cell_pos_to_refresh)
        else:
            return self.cell_assignment(cell_pos_to_refresh)
        # Logger.log("1. Total cost {}, matrix:\r\n".format(np.sum(cost_data)))
        # Logger.log(str(cost_data))
        # return malicious_number, normal_number, malicious_aoi, normal_aoi


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

    def worker_bid_refresh(self):
        for work in self.__worker:
            work.bid_refresh()

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
        malicious_number, normal_number, malicious_assignment, normal_assignment, cost_data, task_assignment_cpu_time = self.cell_step(refresh_cells)        # 小区进行任务分配和执行
        Logger.log("1. Total cost {}, matrix:\r\n".format(np.sum(cost_data)))
        Logger.log(str(cost_data))
        done = False

        # prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, \
        # prev_position, next_position, prev_energy, next_energy, \
        # prev_charge_state, next_charge_state, uav_action_index = self.uav_step()  # 根据训练/测试方式，选择按观测aoi/实际aoi作为输入

        prev_state, action, action_values, next_state, uav_decision_time = self.uav_step()
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


        if self.__train and not self.no_uav:
            self.__agent.memorize(prev_state, action, next_state, reward, done)

        self.__episode_real_aoi.append(np.sum(next_state.real_aoi))
        self.__episode_observation_aoi.append(np.sum(next_state.observation_aoi))
        self.__episode_energy.append(next_state.energy)
        self.__episode_reward.append(reward)

        self.worker_trust_refresh()  # worker信任刷新

        if not self.__train:
            good_trust = []
            bad_trust = []
            action_tuple = (prev_state.position[0], prev_state.position[1], next_state.position[0], next_state.position[1])
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

            act_values = self.__episode_data['move_action'][self.__episode - 1].get(action_tuple, 0)
            self.__episode_data['move_action'][self.__episode - 1][action_tuple] = act_values + 1
            self.__episode_data['uav_move_decision_time'][self.__episode - 1, self.__current_slot - 1] = uav_decision_time
            self.__episode_data['task_assignment_time'][self.__episode - 1, self.__current_slot - 1] = task_assignment_cpu_time
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
                     actual_slot=self.__episode_data['actual_slot'],
                     move_action=self.__episode_data['move_action'],
                     uav_move_decision_time=self.__episode_data['uav_move_decision_time'],
                     task_assignment_time=self.__episode_data['task_assignment_time'])

    def episode_clear(self):
        # 刷新状态，重新开始
        for row_cells in self.__cell:
            for cell in row_cells:
                cell.episode_clear()
        np.random.seed(self.__episode)
        malicious = np.random.random(size=(10357,)) < self.mali_rate
        np.random.seed(self.__episode)
        # random_samples = np.random.choice(list(range(10357)), 3, replace=False)
        for idx, work in enumerate(self.__worker):
            work.episode_clear(malicious[idx], self.cleaner.worker_position[idx])
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
            if self.__episode >= 400 and self.__episode % 25 == 0:
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

