from cell_model import uniform_generator
from worker_model import UAV, Worker, MobilePolicy
from random import randint
from global_parameter import Global as g
from agent_model import DQNAgent
import numpy as np
from energy_model import Energy
from persistent_model import Persistent
from logger import Logger
import os


class Environment:
    def __init__(self, console_log: bool, file_log: bool, train: bool, continue_train: bool):
        self.__cell_limit = g.cell_limit
        self.__charge_cells = g.charge_cells
        self.__uav_fix = g.uav_start_fix
        self.__uav_start_location = g.uav_start_location
        self.__sensor_number = g.sensor_number
        action_size = 7 if g.map_style == 'h' else 5
        self.__agent = DQNAgent(self.__cell_limit, action_size, )
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
        self.__train = train                    # 训练模式
        self.__continue_train = continue_train  # 续训模式
        self.__persistent = Persistent(self.__save_path, self.__train, self.__continue_train)
        self.__logger = Logger(self.__save_path, console_log, file_log)

        Energy.init()

        # network model
        self.__cell = uniform_generator()
        self.__uav = UAV(randint(0, self.__cell_limit - 1), randint(0, self.__cell_limit - 1))
        self.__worker = [Worker(randint(0,  self.__cell_limit - 1), randint(0,  self.__cell_limit - 1),
                                g.worker_initial_trust, g.out_able, g.worker_work_rate)
                         for _ in range(g.worker_number)]

        self.__sec_per_slot = g.sec_per_slot

    def get_cell_observation_aoi(self, current_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_observation_aoi(current_slot)
        ret /= self.__sensor_number # 平均aoi？
        return ret

    def get_cell_real_aoi(self, current_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_real_aoi(current_slot)
        ret /= self.__sensor_number
        return ret

    def get_analyze(self):
        cell_aoi = sum(sum(self.get_cell_real_aoi(self.__current_slot)))
        uav_location = self.get_position_state()
        return cell_aoi, uav_location

    def get_uav_energy_state(self):
        ret = np.empty((1,), dtype=np.float64)
        ret[0] = self.__uav.get_energy() / self.__uav.max_energy
        assert ret[0] <= 1.0
        return ret

    def get_position_state(self):
        dx_dy = self.__uav.get_location()
        assert (len(dx_dy) == 2 and dx_dy[0] >= 0 and dx_dy[1] < self.__cell_limit)
        return np.array(dx_dy)

    def get_charge_state(self):
        ret = np.empty((1,), dtype=np.float64)
        ret[0] = 1.0 if self.__uav.get_charge_state else 0.0
        return ret

    def get_network_state(self):
        # 获取网络当前观测aoi/实际aoi/无人机位置/能量信息
        observation_aoi = self.get_cell_observation_aoi(self.__current_slot)   # 观测aoi
        real_aoi = self.get_cell_real_aoi(self.__current_slot)                 # 实际aoi
        position = self.get_position_state()                                   # uav位置
        energy = self.get_uav_energy_state()                                   # 能量
        charge_state = self.get_charge_state()
        return observation_aoi, real_aoi, position, energy, charge_state

    def uav_step(self):
        # uav步进
        prev_observation_aoi, prev_real_aoi, prev_position, prev_energy, prev_charge_state = self.get_network_state()
        aoi_state = prev_real_aoi if self.__train else prev_observation_aoi
        # 根据上述状态，利用神经网络寻找最佳动作
        uav_action_index = self.__agent.act(aoi_state, prev_position, prev_energy, prev_charge_state)
        # 将index转换为二维方向dx_dy
        uav_action = MobilePolicy.get_action(uav_action_index, g.map_style)
        # 对无人机的状态进行更新, True: 充电，False: 不充电
        next_charge_state = self.__uav.act(uav_action)
        # 获取下一个位置信息
        next_position = self.get_position_state()

        # hover = True if charge_state or ((prev_position == next_position).all()) else False   # 悬浮状态需要计算

        # if charge_state:  这里原意是如果进入了充电，将耗费几个slot，新的模型只需要耗掉1个slot即可
        #     self.__charge_slot = self.__slot_for_charge
        #     hover = False

        self.__cell[next_position[0]][next_position[1]].uav_visited(self.__current_slot)    # 小区更新，无人机访问

        next_observation_aoi = self.get_cell_observation_aoi(self.__current_slot)
        next_real_aoi = self.get_cell_real_aoi(self.__current_slot)
        next_energy = self.get_uav_energy_state()

        return prev_observation_aoi, next_observation_aoi, \
            prev_real_aoi, next_real_aoi, \
            prev_position, next_position, \
            prev_energy, next_energy, \
            prev_charge_state, next_charge_state, \
            uav_action_index

    def reward_calculate(self,
                         prev_aoi, next_aoi, prev_position, next_position,
                         prev_energy, next_energy, prev_charge_state, next_charge_state):
        # reward 模型，可能后续有更改
        # punish = self.__punish if next_energy <= 0 else 0
        # 因为这里是训练得到的reward，因此用real_aoi进行计算
        # 这里虽然无人机可能会花费几个slot来换电池，但是我们对于模型的预测仍然采用下一个时隙的结果进行预测
        # To do: 惩罚因子仍旧有些问题，尝试一些方法解决权重相关的问题
        # reward = - np.sum(next_real_aoi) - punish - self.__hover_punish * hover
        hover_punish = True if (next_position == prev_position and next_position not in self.__charge_cells) else False
        reward = - np.sum(next_aoi) - self.__punish - self.__hover_punish * hover_punish
        return reward

    def workers_step(self):
        cell_pos_to_refresh = set()
        for worker in self.__worker:
            [x, y] = worker.move()
            if worker.work(self.__cell[x][y]):
                cell_pos_to_refresh.add((x, y))
        for tup in cell_pos_to_refresh:
            self.__cell[tup[0]][tup[1]].worker_visited(self.__current_slot)

    def worker_trust_refresh(self):
        for work in self.__worker:
            work.update_trust()

    def slot_step(self):
        # 整合test和train的slot步进方法
        self.workers_step()  # worker先行移动
        done = False

        prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, \
            prev_position, next_position, prev_energy, next_energy, \
            prev_charge_state, next_charge_state, uav_action_index = self.uav_step()  # 根据训练/测试方式，选择按观测aoi/实际aoi作为输入

        if self.__current_slot >= self.__max_slot or next_energy <= 0.0:  # 这里需要考虑电量问题?电量不足时是否直接结束状态
            done = True

        reward = self.reward_calculate(prev_real_aoi, next_real_aoi,
                                       prev_position, next_position,
                                       prev_energy, next_energy,
                                       prev_charge_state, next_charge_state)

        self.__agent.memorize(prev_observation_aoi, next_observation_aoi,
                              prev_real_aoi, next_real_aoi,
                              prev_position, next_position,
                              prev_energy, next_energy,
                              prev_charge_state, next_charge_state,
                              uav_action_index, reward, done)

        self.__persistent.save_data(episode=self.__episode,
                                    slot=self.__current_slot,
                                    real_aoi=next_real_aoi,
                                    observation_aoi=next_observation_aoi,
                                    uav_position=next_position,
                                    reward=reward,
                                    energy=next_energy,
                                    epsilon=self.__agent.epsilon)

        self.worker_trust_refresh()     # worker信任刷新

        if self.__sum_slot % 100 == 0:
            self.__agent.update_target_model()

        if len(self.__agent.memory) > self.__batch_size:
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
            if self.slot_step():
                break
        self.clear()
        self.__agent.save(self.__persistent.model_path())

    def start(self):
        for episode in range(1, self.__max_episode + 1):
            self.__episode = episode
            self.episode_step()
