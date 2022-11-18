from random import random as rand
from random import choice, sample, randint
from global_parameter import Global as g
from energy_model import Energy


class MobilePolicy:
    __hexagon_policy = [[1, 1], [-1, -1], [1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
    __grid_policy = [[0, 1], [1, 0], [-1, 0], [0, -1], [0, 0]]

    # 蜂窝小区随机动作模型
    @staticmethod
    def hexagon_policy():
        return choice(MobilePolicy.__hexagon_policy)

    # 格点小区随机动作模型
    @staticmethod
    def grid_policy():
        return choice(MobilePolicy.__grid_policy)

    # 根据索引选择动作模型
    @staticmethod
    def get_action(index: int, map_style: str):
        if map_style == 'h':
            return MobilePolicy.__hexagon_policy[index]
        elif map_style == 'g':
            return MobilePolicy.__grid_policy[index]
        else:
            assert False

    @staticmethod
    def random_choice(map_style: str):
        if map_style == 'h':
            return MobilePolicy.hexagon_policy()
        elif map_style == 'g':
            return MobilePolicy.grid_policy()
        else:
            assert False


class WorkerBase:

    def __init__(self, x_start: int, y_start: int, initial_trust: float, out_able: bool, fix_start: bool):
        self._x_start = x_start
        self._y_start = y_start
        self._x = x_start
        self._y = y_start
        self._initial_trust = initial_trust
        self._trust = initial_trust
        self._out_able = out_able
        self._fix_start = fix_start

    def action(self, dx_dy):
        [dx, dy] = dx_dy
        if dx == 0 and dy == 0:
            return self._x, self._y

        if not self._out_able:          # 不可移出边界时，只能卡在边界处
            self._x = min(g.cell_limit - 1, max(0, self._x + dx))
            self._y = min(g.cell_limit - 1, max(0, self._y + dy))
        else:
            self._x = (self._x + dx) % g.cell_limit
            self._y = (self._y + dy) % g.cell_limit
        return self._x, self._y

    @property
    def trust(self):
        return self._trust

    @property
    def position(self):
        return [self._x, self._y]

    def clear(self):
        if self._fix_start:
            self._x = self._x_start
            self._y = self._y_start
        else:
            self._x = randint(0,  g.cell_limit - 1)
            self._y = randint(0,  g.cell_limit - 1)
        self._trust = self._initial_trust


class Worker(WorkerBase):
    __index = 0

    def __init__(self,
                 x_start: int,
                 y_start: int,
                 worker_initial_trust: float,
                 out_able: bool,
                 work_rate: float
                 ):
        super(Worker, self).__init__(x_start, y_start, worker_initial_trust, out_able, g.worker_start_fix)
        self.id = Worker.__index
        Worker.__index += 1
        self._work_rate = work_rate
        self._honest = 1    # ?
        self._success_cnt = 0
        self._fail_cnt = 0

    def clear(self):
        super(Worker, self).clear()
        self._success_cnt = 0
        self._fail_cnt = 0

    @staticmethod
    def worker_move_model():
        # worker节点的随机移动模型，后续可能有改动
        return MobilePolicy.random_choice(g.map_style)

    def get_honest(self):
        return self._honest

    def move(self):
        return self.action(Worker.worker_move_model())

    def work(self, cell):
        # work进行移动和工作，随机采集一些节点内的数据，采集后进行上报。当然其采集的性能是依赖于自身honest属性。
        if rand() > self._work_rate:  # 假设工作率是80%，表明在本slot内，80%会执行采集任务
            return False
        sensors = cell.get_sensors()
        selected_sensors = sample(sensors, randint(0, len(sensors)))
        if len(selected_sensors) > 0:
            for sensor in selected_sensors:
                sensor.add_worker(self)
            return True
        else:
            return False

    # 信任更新模型 这里后续可以修改
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
        super(UAV, self).__init__(x_start, y_start, 1.0, False, g.uav_start_fix)
        self.__max_energy = g.uav_energy
        self.__energy = g.uav_energy
        self.__charge_cells = g.charge_cells                 # 可充电小区
        self.__charge_everywhere = g.charge_everywhere       # 任意位置充电
        self.__sec_per_slot = g.sec_per_slot                 # 每个时隙代表多少秒
        self.__charge_state = False                         # 标记是否正在充电
        # ceil(g["charge_time"] / self._sec_per_slot)

    def act(self, dx_dy):
        prev_location = self.position
        # 检测是否主动作出了悬浮操作，如果是，则检查是否悬浮在可充电区域上方，是则充电；如果配置了任意单元可充电，则不需要检查充电区域
        if dx_dy == [0, 0] and (self.__charge_everywhere or (prev_location in self.__charge_cells)):
            self.__charge()
            return

        new_location = super(UAV, self).action(dx_dy)
        self.__energy_consumption(prev_location == new_location)

    @property
    def energy(self) -> float:
        return self.__energy

    @property
    def max_energy(self) -> float:
        return self.__max_energy

    def __charge(self):
        # self._energy = self.max_energy
        self.__charge_state = True
        self._energy = max(self.max_energy, self.__energy + Energy.charge_energy_one_slot())

    def __energy_consumption(self, move: bool):
        self.__charge_state = False
        self.__energy -= (Energy.move_energy_cost() if move else Energy.hover_energy_cost())

    def clear(self):
        super(UAV, self).clear()            # 位置清除
        self.__energy = self.max_energy     # 能量恢复
        self.__charge_state = False
        # 信任不需要变化，uav信任保持为1

    @property
    def get_charge_state(self):
        return self.__charge_state
