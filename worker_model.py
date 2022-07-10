from random import random as rand
from random import choice, sample, randint
from global_ import Global
from energy_model import Energy
from math import ceil, sqrt
class MobilePolicy:
    __hexagon_policy = [[1, 1], [-1, -1], [1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
    __grid_policy = [[0, 1], [1, 0], [-1, 0], [0, -1], [0, 0]]

    @staticmethod
    def hexagon_policy():
        return choice(MobilePolicy.__hexagon_policy)

    @staticmethod
    def grid_policy():
        return choice(MobilePolicy.__grid_policy)

    @staticmethod
    def get_action(index: int):
        g = Global()
        return MobilePolicy.__hexagon_policy[index] if g["map_style"] == 'h' else MobilePolicy.__grid_policy[index]

class WorkerBase:
    def __init__(self, x_start: int, y_start: int, initial_trust: float, out_able: bool):
        self._x = x_start
        self._y = y_start
        self._trust = initial_trust
        self._out_able = out_able

        g = Global()
        self._map_style = g["map_style"]  # g for grid, h for hexagon
        self._cell_limit = g["cell_limit"]

    def action(self, dx_dy):
        [dx, dy] = dx_dy
        if not self._out_able:
            self._x = min(self._cell_limit - 1, max(0, self._x + dx))
            self._y = min(self._cell_limit - 1, max(0, self._y + dy))
        else:
            self._x = (self._x + dx) % self._cell_limit
            self._y = (self._y + dy) % self._cell_limit
        return self._x, self._y

    def get_trust(self):
        return self._trust

    def get_location(self):
        return [self._x, self._y]


class Worker(WorkerBase):
    __index = 0

    def __init__(self, x_start: int, y_start: int):
        g = Global()
        super(Worker, self).__init__(x_start, y_start, g["worker_initial_trust"], g["out_able"])
        self.id = Worker.__index
        Worker.__index += 1
        self._activity = g["worker_activity"]
        self._honest = 1
        self._success_cnt = 0
        self._fail_cnt = 0

    def clear(self, x_start, y_start, initial_trust):
        self._trust = initial_trust
        self._success_cnt = 0
        self._fail_cnt = 0
        self._x = x_start
        self._y = y_start

    def __worker_move_model(self):
        # worker节点的移动模型
        if rand() < self._activity:
            if self._map_style == 'h':
                return MobilePolicy.hexagon_policy()
            elif self._map_style == 'g':
                return MobilePolicy.grid_policy()
            else:
                assert False
        return [0, 0]

    def get_honest(self):
        return self._honest

    def move(self):
        return self.action(self.__worker_move_model())

    def work(self, cell):
        # work进行移动和工作，随机采集一些节点内的数据，采集后进行上报。当然其采集的性能是依赖于自身honest属性。
        sensors = cell.get_sensors()
        selected_sensors = sample(sensors, randint(0, len(sensors)))
        if len(selected_sensors) > 0:
            for sensor in selected_sensors:
                sensor.add_worker(self)
            return True
        else:
            return False

    def update_trust(self):
        s = self._success_cnt / (self._success_cnt + self._fail_cnt + 1)
        f = 1 / (self._success_cnt + self._fail_cnt + 1)
        self._trust = (2 * s + f) / 2

    def add_success(self, cnt):
        self._success_cnt += cnt

    def add_fail(self, cnt):
        self._fail_cnt += cnt


class UAV(WorkerBase):
    def __init__(self, x_start, y_start):
        super(UAV, self).__init__(x_start, y_start, 1.0, False)
        g = Global()
        self.max_energy = g["uav_energy"]
        self._energy = self.max_energy
        self._charge_cells = g["charge_cells"]
        self._slot_step_for_charge = 4     #用于表示充电所耗费的时隙数量
        self._sec_per_slot = g["cell_length"] * sqrt(3) / g["uav_speed"] \
            if g["map_style"] == 'h' else g["cell_length"] / g["uav_speed"]
        self._slot_for_charge = ceil(g["charge_time"] / self._sec_per_slot)

    def action(self, dx_dy):
        prev_location = self.get_location()
        if dx_dy == [0, 0] and prev_location in self._charge_cells:
            self.__charge()
            return True     # charge ?

        super(UAV, self).action(dx_dy)
        new_location = self.get_location()
        if prev_location == new_location:
            self._energy -= (Energy.hover_energy_cost()  * self._sec_per_slot / 3600)# 到达了边界无法移动，仅能进行悬浮操作
        else:
            self._energy -= (Energy.move_energy_cost() * self._sec_per_slot / 3600)
        return False

    def get_energy(self):
        return self._energy

    def __charge(self):
        self._energy = self.max_energy

    def clear(self, x_start, y_start):
        self._energy = self.max_energy
        self._x = x_start
        self._y = y_start

    def get_second_per_slot(self):
        return self._sec_per_slot

    def get_charge_slot(self):
        return self._slot_for_charge