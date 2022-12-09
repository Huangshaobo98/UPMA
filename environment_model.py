# from sensor_model import Sensor
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


class Environment:
    def __init__(self,
                 train: bool,
                 continue_train: bool,
                 sensor_number: int,
                 worker_number: int,
                 max_episode: int,
                 # max_slot: int,
                 batch_size: int,
                 epsilon_decay: float,
                 learn_rate: float,
                 gamma: float,
                 detail: bool,
                 cleaner: DataCleaner
                 ):

        # 一些比较固定的参数
        # self.__charge_cells = g.charge_cells
        # self.__uav_fix = g.uav_start_fix
        # self.__uav_start_location = g.uav_start_location
        # self.__initial_trust = g.initial_trust

        self.__train = train  # 训练模式
        # self.__continue_train = continue_train  # 续训模式

        # self.__sensor_number = sensor_number
        # self.__worker_number = worker_number

        self.cleaner = cleaner

        self.__detail = detail

        # 训练相关
        self.__max_episode = max_episode if train else 1
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

        # energy
        Energy.init(cleaner)

        # agent
        self.__agent = DQNAgent(cleaner.cell_limit, action_size=7, gamma=gamma, epsilon=Persistent.trained_epsilon(),
                                epsilon_decay=epsilon_decay, lr=learn_rate, train=train, continue_train=continue_train,
                                model_path=Persistent.model_path())

        # network model
        self.__cell = Cell.uniform_generator_with_position(cleaner.x_limit,
                                                           cleaner.y_limit,
                                                           cleaner.cell_coordinate,
                                                           sensor_number,
                                                           cleaner.side_length)
        WorkerBase.set_cleaner(cleaner)
        self.__uav = UAV(cleaner.cell_limit)
        self.__worker = [Worker(i, cleaner.worker_position[i]) for i in range(worker_number)]

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
        reward = - np.sum(next_state.real_aoi_state) \
                 + self.__no_power_punish * g.energy_reward_calculate(next_state.energy_state[0]) \
                 - self.__hover_punish * (hover and not charge)

        return reward

    def uav_step(self):
        # uav步进
        Logger.log("\r\n" + "-" * 36 + " UAV step. " + "-" * 36)
        prev_state = self.get_current_network_state()
        # 根据上述状态，利用神经网络寻找最佳动作
        uav_action_index, action_values = self.__agent.act(prev_state)
        # 将index转换为二维方向dx_dy
        uav_action = MobilePolicy.get_action(uav_action_index)
        # 无人机移动，主要是动作改变和充电的工作
        self.__uav.act(uav_action)
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
        for x, y in cell_pos_to_refresh:
            self.__cell[x, y].task_assignment(self.__current_slot)    # 任务分配
            self.__cell[x, y].worker_visited(self.__current_slot)     # 任务执行

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
            work.update_trust()

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
        self.cell_step(refresh_cells)        # 小区进行任务分配和执行

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
                                              reward, self.__agent.epsilon))
        if self.__train:
            self.__agent.memorize(prev_state, action, next_state, reward, done)

        self.__episode_real_aoi.append(np.sum(next_state.real_aoi))
        self.__episode_observation_aoi.append(np.sum(next_state.observation_aoi))
        self.__episode_energy.append(next_state.energy)
        self.__episode_reward.append(reward)

        persist_data = {
            'episode': self.__episode,
            'slot': self.__current_slot,
            'sum real aoi': np.sum(next_state.real_aoi),
            'average real aoi': np.average(next_state.real_aoi),
            'sum observation aoi': np.sum(next_state.observation_aoi),
            'average observation aoi': np.average(next_state.observation_aoi),
            'hover': hover,
            'charge': self.get_charge_state(),
            'uav position x': next_state.position[0],
            'uav position y': next_state.position[1],
            'reward': reward,
            'energy': next_state.energy,
            'energy left rate': next_state.energy_state.sum(),
            'epsilon': self.__agent.epsilon
        }
        Persistent.save_data(persist_data)

        self.worker_trust_refresh()  # worker信任刷新

        if self.__train and self.__sum_slot % 100 == 0:
            self.__agent.update_target_model()

        if self.__train and len(self.__agent.memory) >= self.__batch_size:
            self.__agent.replay(self.__batch_size)

        return done

    def episode_clear(self):
        # 刷新状态，重新开始
        for row_cells in self.__cell:
            for cell in row_cells:
                cell.episode_clear()
        for work in self.__worker:
            work.episode_clear()
        self.__uav.clear()
        self.__episode_real_aoi.clear()
        self.__episode_observation_aoi.clear()
        self.__episode_energy.clear()
        self.__episode_reward.clear()

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
            'slot number': self.__current_slot
        }
        Persistent.save_episode_data(episode_data)
        self.episode_clear()
        self.__agent.save(Persistent.model_path())

    def start(self):
        np.set_printoptions(suppress=True, precision=3)
        for episode in range(Persistent.trained_episode() + 1, self.__max_episode + 1):
            self.__episode = episode
            self.episode_step()
