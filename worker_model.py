from random import random as rand
from random import choice, sample, randint
from global_parameter import Global as g
from energy_model import Energy
from logger import Logger
from data.data_clean import DataCleaner


class MobilePolicy:
    __hexagon_policy = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 0], [0, 0]]
    # __grid_policy = [[0, 1], [1, 0], [-1, 0], [0, -1], [0, 0]]

    # 蜂窝小区随机动作模型
    @staticmethod
    def hexagon_policy():
        return choice(MobilePolicy.__hexagon_policy)

    # # 格点小区随机动作模型
    # @staticmethod
    # def grid_policy():
    #     return choice(MobilePolicy.__grid_policy)

    # 根据索引选择动作模型
    @staticmethod
    def get_action(index: int,
                   # map_style: str
                   ):
        # if map_style == 'h':
        return MobilePolicy.__hexagon_policy[index]
        # elif map_style == 'g':
        #     return MobilePolicy.__grid_policy[index]
        # else:
        #     assert False

    @staticmethod
    def random_choice(map_style: str):
        # if map_style == 'h':
        return MobilePolicy.hexagon_policy()
        # elif map_style == 'g':
        #     return MobilePolicy.grid_policy()
        # else:
        #     assert False


class WorkerBase:
    cleaner = None

    @staticmethod
    def set_cleaner(cleaner: DataCleaner):
        WorkerBase.cleaner = cleaner

    def __init__(self,
                 # x_start: int, y_start: int,
                 initial_trust: float,
                 # out_able: bool, fix_start: bool
                 ):
        # self._x_start = x_start
        # self._y_start = y_start
        # self._x = x_start
        # self._y = y_start
        self.initial_trust = initial_trust
        self.trust = initial_trust
        # self._out_able = out_able
        # self._fix_start = fix_start

    # def action(self, dx_dy):
    #     [dx, dy] = dx_dy
    #     if dx == 0 and dy == 0:
    #         return self._x, self._y
    #
    #     if not self._out_able:          # 不可移出边界时，只能卡在边界处
    #         self._x = min(WorkerBase.cell_limit - 1, max(0, self._x + dx))
    #         self._y = min(WorkerBase.cell_limit - 1, max(0, self._y + dy))
    #     else:
    #         self._x = (self._x + dx) % WorkerBase.cell_limit
    #         self._y = (self._y + dy) % WorkerBase.cell_limit
    #     return self._x, self._y


    def clear(self):
        # if self._fix_start:
        #     self._x = self._x_start
        #     self._y = self._y_start
        # else:
        #     self._x = randint(0,  WorkerBase.cell_limit - 1)
        #     self._y = randint(0,  WorkerBase.cell_limit - 1)
        self.trust = self.initial_trust


class Worker(WorkerBase):

    def __init__(self,
                 index,
                 # x_start: int,
                 # y_start: int,
                 positions: dict,
                 worker_initial_trust: float = g.initial_trust,
                 # out_able: bool,
                 # work_rate: float = g.worker_work_rate,
                 vitality: int = g.worker_vitality,
                 direct_window: int = 10,
                 recom_window: int = 20,
                 ):
        super(Worker, self).__init__(worker_initial_trust)
        self._id = index
        self._work_position = positions
        # self._work_rate = work_rate
        self._honest = 1    # ?
        self._vitality = vitality   # 活性，指的是每个时隙最多采集多少个传感器节点
        self._direct_trust = self.initial_trust
        self._direct_trust_fresh = False
        self._temp_direct_trust_success = 0
        self._temp_direct_trust_fail = 0
        self._direct_trust_success = []
        self._direct_trust_fail = []
        self._direct_limit = direct_window

        self._weight = 0.7

        self._recom_trust = self.initial_trust
        self._recom_trust_fresh = False
        self._recom_trust_list = []
        self._recom_result_list = []
        self._recom_limit = recom_window
        # self._success_cnt = 0
        # self._fail_cnt = 0

    def episode_clear(self):
        super(Worker, self).clear()
        self._direct_trust = self.initial_trust
        self._direct_trust_fresh = False
        self._temp_direct_trust_success = 0
        self._temp_direct_trust_fail = 0
        self._direct_trust_success = []
        self._direct_trust_fail = []
        self._direct_limit = 10


        self._recom_trust = self.initial_trust
        self._recom_trust_fresh = False
        self._recom_trust_list = []
        self._recom_result_list = []
        self._recom_limit = 20

    # @staticmethod
    # def worker_move_model():
    #     # worker节点的随机移动模型，后续可能有改动
    #     return MobilePolicy.random_choice(g.map_style)

    @property
    def vitality(self):
        return self._vitality

    @property
    def index(self):
        return self._id

    @property
    def honest(self):
        return self._honest

    def move(self, current_slot):
        if self._work_position is None:
            return None
        return self._work_position[current_slot] if current_slot in self._work_position else None

    # def work(self, cell):
    #     # work进行移动和工作，随机采集一些节点内的数据，采集后进行上报。当然其采集的性能是依赖于自身honest属性。
    #     if rand() > self._work_rate:  # 假设工作率是80%，表明在本slot内，80%会执行采集任务
    #         return False
    #     selected_sensors = sample(cell.sensors, randint(0, int(cell.sensor_number * self._activity)))
    #
    #     sensor_index = [sensor.index for sensor in selected_sensors]
    #     Logger.log("Worker {} sampled at cell {}, sampled number: {}, sampled sensor list: {}"
    #                .format(self.index, cell.index, len(sensor_index), sensor_index))
    #     if len(selected_sensors) > 0:
    #         for sensor in selected_sensors:
    #             sensor.add_worker(self)
    #         return True
    #     else:
    #         return False

    # 信任更新模型 这里后续可以修改
    def update_trust(self):
        if self._direct_trust_fresh:
            assert len(self._direct_trust_success) == len(self._direct_trust_fail)
            self._direct_trust_success.append(self._temp_direct_trust_success)
            self._direct_trust_fail.append(self._temp_direct_trust_fail)
            if len(self._direct_trust_success) > self._direct_limit:
                self._direct_trust_success.pop(0)
            if len(self._direct_trust_fail) > self._direct_limit:
                self._direct_trust_fail.pop(0)
            self._direct_trust = 0.5 * (2 * sum(self._direct_trust_success) + 1) \
                                 / (sum(self._direct_trust_success) + sum(self._direct_trust_fail) + 1)
        if self._recom_trust_fresh:
            assert len(self._recom_trust_list) == len(self._recom_result_list)
            while len(self._recom_trust_list) > self._recom_limit:
                self._recom_trust_list.pop(0)
            while len(self._recom_result_list) > self._recom_limit:
                self._recom_result_list.pop(0)

            self._recom_trust = sum([i*j for i, j in zip(self._recom_trust_list, self._recom_result_list)]) \
                                / sum(self._recom_trust_list)

        self.trust = self._direct_trust * self._weight + self._recom_trust * (1 - self._weight)
        self._direct_trust_fresh = False
        self._recom_trust_fresh = False
        self._temp_direct_trust_success = 0
        self._temp_direct_trust_fail = 0

        return [self.trust, self._direct_trust, self._recom_trust]

    def add_direct_success(self, cnt):
        self._direct_trust_fresh = True
        self._temp_direct_trust_success += cnt

    def add_direct_fail(self, cnt):
        self._direct_trust_fresh = True
        self._temp_direct_trust_fail += cnt

    def add_recom_result(self, worker, result:bool):
        if self == worker:
            return
        self._recom_trust_fresh = True
        self._recom_trust_list.append(worker.trust)
        self._recom_result_list.append(1.0 if result else 0.0)


class UAV(WorkerBase):
    def __init__(self, cell_limit):
        super(UAV, self).__init__(1.0)
        [self.__x, self.__y] = [self.__x_start, self.__y_start] = g.uav_start_location
        self.__max_energy = g.uav_energy
        self.__energy = g.uav_energy
        [self.__x_limit, self.__y_limit] = cell_limit
        # self.__charge_cells = g.charge_cells                 # 可充电小区
        # self.__sec_per_slot = g.sec_per_slot                 # 每个时隙代表多少秒
        self.__charge_state = False                         # 标记是否正在充电
        # ceil(g["charge_time"] / self._sec_per_slot)

    @property
    def position(self):
        return [self.__x, self.__y]

    # def action(self, dx_dy):
    #     [dx, dy] = dx_dy
    #     if dx == 0 and dy == 0:
    #         return self._x, self._y
    #
    #     if not self._out_able:          # 不可移出边界时，只能卡在边界处
    #         self._x = min(WorkerBase.cell_limit - 1, max(0, self._x + dx))
    #         self._y = min(WorkerBase.cell_limit - 1, max(0, self._y + dy))
    #     else:
    #         self._x = (self._x + dx) % WorkerBase.cell_limit
    #         self._y = (self._y + dy) % WorkerBase.cell_limit
    #     return self._x, self._y

    def act(self, dx_dy):
        prev_location = self.position
        # 检测是否主动作出了悬浮操作，如果是，则检查是否悬浮在可充电区域上方，是则充电；如果配置了任意单元可充电，则不需要检查充电区域
        if dx_dy == [0, 0]:
            if g.charge_everywhere or (not g.charge_everywhere and (prev_location in g.charge_cells)):
                self.__charge()
                return
        [dx, dy] = dx_dy
        new_location = [self.__x, self.__y] \
                     = [min(self.__x_limit - 1, max(0, self.__x + dx)),
                        min(self.__y_limit - 1, max(0, self.__y + dy))]

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
        self.__energy = min(self.max_energy, self.__energy + Energy.charge_energy_one_slot())

    def __energy_consumption(self, move: bool):
        self.__charge_state = False
        self.__energy -= (Energy.move_energy_cost() if move else Energy.hover_energy_cost())

    def clear(self):
        super(UAV, self).clear()            # 位置清除
        self.__energy = self.max_energy     # 能量恢复
        self.__charge_state = False
        self.__x = self.__x_start
        self.__y = self.__y_start
        # 信任不需要变化，uav信任保持为1

    @property
    def charge_state(self) -> bool:
        return self.__charge_state
