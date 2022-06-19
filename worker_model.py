from random import random as rand
from random import choice
from global_ import Global
from energy_model import Energy


class MobilePolicy:
    __hexagon_policy = [[1, 1], [-1, -1], [1, 0], [0, 1], [-1, 0], [0, -1]]
    __grid_policy = [[0, 1], [1, 0], [-1, 0], [0, -1]]

    @staticmethod
    def hexagon_policy():
        return choice(MobilePolicy.__hexagon_policy)

    @staticmethod
    def grid_policy():
        return choice(MobilePolicy.__grid_policy)


class WorkerBase:
    def __init__(self, x_start: int, y_start: int, initial_trust: float, out_able: bool):
        self.__x = x_start
        self.__y = y_start
        self.__trust = initial_trust
        self.__out_able = out_able

        g = Global()
        self.__map_style = g["map_style"]  # g for grid, h for hexagon
        self.__x_limit = g["x_limit"]
        self.__y_limit = g["y_limit"]

    def action(self, dx_dy):
        [dx, dy] = dx_dy
        if not self.__out_able:
            self.__x = min(self.__x_limit, max(0, self.__x + dx))
            self.__y = min(self.__y_limit, max(0, self.__y + dy))
        else:
            self.__x = (self.__x + dx) % self.__x_limit
            self.__y = (self.__y + dy) % self.__y_limit

    def get_trust(self):
        return self.__trust

    def get_location(self):
        return self.__x, self.__y


class Worker(WorkerBase):
    __index = 0

    def __init__(self, x_start: int, y_start: int):
        g = Global()
        super(Worker, self).__init__(x_start, y_start, g["worker_initial_trust"], g["out_able"])
        self.id = Worker.__index
        Worker.__index += 1
        self.__activity = g["worker_activity"]
        self.__honest = 1
        self.__legal_action_time = 0
        self.__illegal_action_time = 0

    def __worker_move_model(self):
        # worker节点的移动模型
        if rand() < self.__activity:
            if self.__map_style == 'h':
                return MobilePolicy.hexagon_policy()
            elif self.__map_style == 'g':
                return MobilePolicy.grid_policy()
            else:
                assert False
        return [0, 0]

    def move(self):
        self.action(self.__worker_move_model())

    def get_honest(self):
        return self.__honest


class UAV(WorkerBase):
    def __init__(self, x_start, y_start):
        super(UAV, self).__init__(x_start, y_start, 1.0, False)
        g = Global()
        self.__energy = g["uav_energy"]

    def action(self, dx_dy):
        prev = self.get_location()
        super(UAV, self).action(dx_dy)
        new = self.get_location()

        if prev == new:
            self.__energy -= Energy.hover_energy_cost()
        else:
            self.__energy -= Energy.move_energy_cost()

    def get_energy(self):
        return self.__energy
