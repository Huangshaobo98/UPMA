from cell_model import uniform_generator
from worker_model import UAV, Worker, MobilePolicy
from random import randint
from global_parameter import Global as g
from agent_model import DQNAgent, State
import numpy as np
from energy_model import Energy
from persistent_model import Persistent
from logger import Logger
from typing import List
import os


class Environment:
    def __init__(self, console_log: bool, file_log: bool, train: bool, continue_train: bool):
        self.__cell_limit = g.cell_limit
        self.__charge_cells = g.charge_cells
        self.__uav_fix = g.uav_start_fix
        self.__uav_start_location = g.uav_start_location
        self.__sensor_number = g.sensor_number
        action_size = 7 if g.map_style == 'h' else 5

        self.__current_slot = 0
        self.__sum_slot = 0
        self.__max_slot = g.max_slot
        self.__punish = g.punish
        self.__hover_punish = g.hover_punish
        self.__batch_size = g.batch_size
        self.__episode = 0
        self.__max_episode = g.max_episode
        self.__initial_trust = g.initial_trust

        self.__save_path = os.getcwd() + "/save/cell_" + str(self.__cell_limit)
        self.__train = train  # 训练模式
        self.__continue_train = continue_train  # 续训模式
        self.__persistent = Persistent(self.__save_path, self.__train, self.__continue_train)
        self.__logger = Logger(self.__save_path, console_log, file_log)

        Energy.init()

        # agent
        self.__agent = DQNAgent(self.__cell_limit, action_size,
                                train=train, continue_train=continue_train, model_path=self.__persistent.model_path())

        # network model
        self.__cell = uniform_generator()
        self.__uav = UAV(randint(0, self.__cell_limit - 1), randint(0, self.__cell_limit - 1))
        self.__worker = [Worker(randint(0, self.__cell_limit - 1), randint(0, self.__cell_limit - 1),
                                g.worker_initial_trust, g.out_able, g.worker_work_rate)
                         for _ in range(g.worker_number)]

        self.__sec_per_slot = g.sec_per_slot

    @property
    def logger(self):
        return self.__logger

    def get_cell_observation_aoi(self, current_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_observation_aoi(current_slot)
        return ret

    def get_cell_real_aoi(self, current_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_real_aoi(current_slot)
        return ret

    def get_analyze(self):
        cell_aoi = sum(sum(self.get_cell_real_aoi(self.__current_slot)))
        uav_location = self.position_state
        return cell_aoi, uav_location

    @property
    def energy_state(self) -> float:
        ret = self.__uav.energy
        return ret

    @property
    def position_state(self) -> List[int]:
        dx_dy = self.__uav.position
        assert (len(dx_dy) == 2 and dx_dy[0] >= 0 and dx_dy[1] < self.__cell_limit)
        return dx_dy

    @property
    def charge_state(self):
        return self.__uav.get_charge_state

    def get_network_state(self) -> State:
        # 获取网络当前观测aoi/实际aoi/无人机位置/能量信息
        observation_aoi = self.get_cell_observation_aoi(self.__current_slot)  # 观测aoi
        real_aoi = self.get_cell_real_aoi(self.__current_slot)  # 实际aoi
        position_state = self.position_state  # uav位置
        energy_state = self.energy_state  # 能量
        charge_state = self.charge_state
        return State(real_aoi, observation_aoi, position_state, energy_state, charge_state)

    def cell_update_by_uav(self):
        uav_position = self.position_state   # 获取无人机的位置信息
        self.__cell[uav_position[0]][uav_position[1]].uav_visited(self.__current_slot)  # 小区更新，无人机访问

    def uav_step(self):
        # uav步进
        self.logger.log("UAV step.")
        prev_state = self.get_network_state()
        # 根据上述状态，利用神经网络寻找最佳动作
        uav_action_index = self.__agent.act(prev_state)
        # 将index转换为二维方向dx_dy
        uav_action = MobilePolicy.get_action(uav_action_index, g.map_style)
        # 无人机移动，主要是动作改变和充电的工作
        self.__uav.act(uav_action)
        # 无人机执行所在小区的数据收集/评估worker的信任等工作
        self.cell_update_by_uav()

        next_state = self.get_network_state()

        # hover = True if charge_state or ((prev_position == next_position).all()) else False   # 悬浮状态需要计算

        # if charge_state:  这里原意是如果进入了充电，将耗费几个slot，新的模型只需要耗掉1个slot即可
        #     self.__charge_slot = self.__slot_for_charge
        #     hover = False

        # next_observation_aoi = self.get_cell_observation_aoi(self.__current_slot)
        # next_real_aoi = self.get_cell_real_aoi(self.__current_slot)
        # next_energy = self.energy_state()

        return prev_state, uav_action_index, next_state

    def reward_calculate(self, prev_state: State, next_state: State, hover: bool, charge: bool, no_power: bool):
        # reward 模型，可能后续有更改
        # punish = self.__punish if next_energy <= 0 else 0
        # 因为这里是训练得到的reward，因此用real_aoi进行计算
        # 这里虽然无人机可能会花费几个slot来换电池，但是我们对于模型的预测仍然采用下一个时隙的结果进行预测
        # To do: 惩罚因子仍旧有些问题，尝试一些方法解决权重相关的问题
        # reward = - np.sum(next_real_aoi) - punish - self.__hover_punish * hover
        # hover and not charge 悬浮但不充电，指的是无意义的悬浮操作
        reward = - np.sum(next_state.real_aoi_state) - self.__punish * no_power\
                 - self.__hover_punish * (hover and not charge)

        return reward

    def workers_step(self):
        self.logger.log("Workers step.")
        cell_pos_to_refresh = set()
        for worker in self.__worker:
            [x, y] = worker.move()
            if worker.work(self.__cell[x][y]):
                cell_pos_to_refresh.add((x, y))
        if len(cell_pos_to_refresh) > 0:
            self.logger.log("Workers work position: {}.".format(cell_pos_to_refresh))
        else:
            self.logger.log("Workers do not work.")
        for tup in cell_pos_to_refresh:
            self.__cell[tup[0]][tup[1]].worker_visited(self.__current_slot)

    def worker_trust_refresh(self):
        for work in self.__worker:
            work.update_trust()

    def slot_step(self):
        # 整合test和train的slot步进方法
        self.workers_step()  # worker先行移动
        done = False

        # prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, \
        # prev_position, next_position, prev_energy, next_energy, \
        # prev_charge_state, next_charge_state, uav_action_index = self.uav_step()  # 根据训练/测试方式，选择按观测aoi/实际aoi作为输入

        prev_state, action, next_state = self.uav_step()
        self.logger.log("State before UAV action:\r\n" + str(prev_state))
        self.logger.log("State after UAV action:\r\n" + str(next_state))
        if self.__current_slot >= self.__max_slot or next_state.energy <= 0.0:  # 这里需要考虑电量问题?电量不足时是否直接结束状态
            done = True

        hover = True if (prev_state.position == next_state.position
                         and next_state.position not in self.__charge_cells) else False
        charge = self.charge_state
        no_power = next_state.energy <= 0.0
        reward = self.reward_calculate(prev_state, next_state, hover, charge, no_power)

        self.logger.log("Reward: {}".format(reward))
        self.__agent.memorize(prev_state, action, next_state, reward, done)

        persist_data = {
            'episode': self.__episode,
            'slot': self.__current_slot,
            'sum real aoi': np.sum(next_state.real_aoi),
            'average real aoi': np.average(next_state.real_aoi),
            'sum observation aoi': np.sum(next_state.observation_aoi),
            'average observation aoi': np.average(next_state.observation_aoi),
            'hover': hover,
            'charge': self.charge_state,
            'uav position': next_state.position,
            'reward': reward,
            'energy': next_state.energy,
            'energy left rate': next_state.energy_state.sum(),
            'epsilon': self.__agent.epsilon
        }
        self.__persistent.save_data(persist_data)

        self.worker_trust_refresh()  # worker信任刷新

        if self.__sum_slot % 100 == 0:
            self.__agent.update_target_model()

        if len(self.__agent.memory) >= self.__batch_size:
            self.__agent.replay(self.__batch_size)

        return done

    def clear(self):
        # 刷新状态，重新开始
        for row in self.__cell:
            for item in row:
                item.clear()
        for work in self.__worker:
            work.clear()
        self.__uav.clear()

    def episode_step(self):
        for slot in range(1, self.__max_slot + 1):
            self.__current_slot = slot
            self.__sum_slot += 1
            self.logger.log("Episode {}, slot {}, sum slot {} begin."
                            .format(self.__episode, self.__current_slot, self.__sum_slot))
            if self.slot_step():
                break
        self.logger.log("===========================Episode {} end.===========================".format(self.__episode))
        self.clear()
        self.__agent.save(self.__persistent.model_path())

    def start(self):
        np.set_printoptions(suppress=True,precision=3)
        for episode in range(1, self.__max_episode + 1):
            self.__episode = episode
            self.episode_step()
