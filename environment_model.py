from cell_model import uniform_generator
from worker_model import UAV, Worker, MobilePolicy
from random import randint
from global_ import Global
from agent_model import DQNAgent
import numpy as np
from energy_model import Energy
from persistent_model import Persistent
import logging
import os

class Environment:
    def __init__(self, console_enable, delete_model, delete_log):
        g = Global()
        self.__cell_limit = g["cell_limit"]
        worker_number = g["worker_number"]
        self.__uav = UAV(randint(0,  self.__cell_limit - 1), randint(0,  self.__cell_limit - 1))
        self.__cell = uniform_generator()
        self.__sensor_number= g["sensor_number"]
        self.__worker = [Worker(randint(0,  self.__cell_limit - 1), randint(0,  self.__cell_limit - 1)) for _ in range(worker_number)]
        self.__sec_per_slot = self.__uav.get_second_per_slot()
        self.__slot_for_charge = self.__uav.get_charge_slot()
        self.__charge_slot = 0
        action_size = 7 if g["map_style"] == 'h' else 5
        self.__agent = DQNAgent(self.__cell_limit, action_size, )
        self.__current_slot = 0
        self.__sum_slot = 0
        self.__max_slot = g["max_slot"]
        self.__punish = g["punish"]
        self.__hover_punish = g["hover_punish"]
        self.__batch_size = g["batch_size"]
        self.__episode = 0
        self.__max_episode = g["max_eposide"]
        self.__initial_trust = g["initial_trust"]
        self.__persistent = Persistent()
        self.__title = os.getcwd() + "/save/cell_" + str(self.__cell_limit)
        self.__model_path = self.__title + "/model.h5"
        self.__log_path = self.__title + "/running.log"
        if not os.path.exists(self.__title):
            os.makedirs(self.__title)
        if delete_model:
            if os.path.exists(self.__model_path):
                os.remove(self.__model_path)
        if delete_log:
            if os.path.exists(self.__log_path):
                os.remove(self.__log_path)

        self.__logger = logging.getLogger("Root")
        self.log_config(console_enable)
        Energy.init()

    def log_config(self, console_enable):
        self.__logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.__title + "/running.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(message)s"))
        if console_enable:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter("%(message)s"))
            self.__logger.addHandler(ch)
        self.__logger.addHandler(fh)

    def get_cell_observation_aoi(self, curret_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_observation_aoi(curret_slot)
        ret /= self.__sensor_number
        return ret

    def get_cell_real_aoi(self, curret_slot):
        ret = np.empty((self.__cell_limit, self.__cell_limit), dtype=np.float64)
        for x in range(self.__cell_limit):
            for y in range(self.__cell_limit):
                ret[x][y] = self.__cell[x][y].get_real_aoi(curret_slot)
        ret /= self.__sensor_number
        return ret

    def get_analyze(self):
        cell_aois = sum(sum(self.get_cell_real_aoi(self.__current_slot)))
        uav_location = self.get_position_state()
        return cell_aois, uav_location

    def get_uav_energy_state(self):
        ret = np.empty((1,), dtype=np.float64)
        ret[0] = self.__uav.get_energy() / self.__uav.max_energy
        return ret

    def get_position_state(self):
        dx_dy = self.__uav.get_location()
        return np.array(dx_dy)

    def uav_step(self, train):
        # ??????: ??????aoi???????????????aoi????????????????????????????????????
        prev_observation_aoi = self.get_cell_observation_aoi(self.__current_slot)
        prev_real_aoi = self.get_cell_real_aoi(self.__current_slot)
        prev_position = self.get_position_state()
        prev_energy = self.get_uav_energy_state()

        uav_action_index = self.__agent.act(prev_real_aoi, prev_observation_aoi, prev_position,
                                            prev_energy, train)

        uav_action = MobilePolicy.get_action(uav_action_index)
        charge_state = self.__uav.action(uav_action) # ?????????????????????????????????

        next_position = self.get_position_state()

        hover = True if (prev_position == next_position).all() else False
        if charge_state:
            self.__charge_slot = self.__slot_for_charge
            hover = False

        self.__cell[next_position[0]][next_position[1]].uav_visited(self.__current_slot)

        next_observation_aoi = self.get_cell_observation_aoi(self.__current_slot)
        next_real_aoi = self.get_cell_real_aoi(self.__current_slot)
        next_energy = self.get_uav_energy_state()

        punish = self.__punish if next_energy <= 0 else 0

        reward = - np.sum(next_real_aoi) - punish - self.__hover_punish * hover

        # ??????????????????????????????????????????slot????????????????????????????????????????????????????????????????????????????????????????????????

        return prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, prev_position, next_position, prev_energy, next_energy, reward, uav_action_index

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

    def slot_step_train(self):
        self.workers_step()                 # worker????????????
        done = False
        if self.__charge_slot <= 0:     # ?????????????????????????????????????????????????????????????????????
            [prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, prev_position, next_position, prev_energy, next_energy, reward, uav_action_index] = self.uav_step(True)
            if self.__current_slot == self.__max_slot:
                done = True
            self.__agent.memorize(prev_observation_aoi, next_observation_aoi, prev_real_aoi, next_real_aoi, prev_position,
                                next_position, prev_energy, next_energy, reward, uav_action_index, done)
            obv_aoi = next_observation_aoi
            real_aoi = next_real_aoi
            uav_pos = next_position
            energy = next_energy
            reward = reward
        else:
            obv_aoi = self.get_cell_observation_aoi(self.__current_slot)
            real_aoi = self.get_cell_real_aoi(self.__current_slot)
            uav_pos = self.get_position_state()
            energy = self.get_uav_energy_state()
            reward = 0

        self.__persistent.print_slot_verbose_1(self.__logger, self.__episode, self.__current_slot, real_aoi,
                                                   obv_aoi, uav_pos, reward, energy, self.__agent.epsilon)

        self.worker_trust_refresh()     # worker????????????
        self.__charge_slot -= 1
        if self.__sum_slot % 100 == 0:
            self.__agent.update_target_model()

        if len(self.__agent.memory) > self.__batch_size:
            self.__agent.replay(self.__batch_size)

        return done


    def clear(self):
        # ???????????????????????????
        for row in self.__cell:
            for item in row:
                item.clear()
        for work in self.__worker:
            work.clear(randint(0,  self.__cell_limit - 1), randint(0,  self.__cell_limit - 1), self.__initial_trust)
        self.__uav.clear(randint(0,  self.__cell_limit - 1), randint(0,  self.__cell_limit - 1))

    def eposide_step(self):
        for slot in range(1, self.__max_slot + 1):
            self.__current_slot = slot
            self.__sum_slot += 1
            if self.slot_step_train():
                break
        self.clear()
        self.__agent.save(self.__model_path)

    def start(self):
        for episode in range(1, self.__max_episode + 1):
            self.__episode = episode
            self.eposide_step()









