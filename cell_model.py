import numpy as np

from sensor_model import Sensor
from numpy import random
from math import sqrt
from data.data_clean import DataCleaner
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, x, y, position, side_length):
        self.__x = x
        self.__y = y
        self.__x_location = position[0]
        self.__y_location = position[1]
        self.__sensors = []
        self.__length = side_length

        # self.__map_style = g.map_style

        # self.__x_location = 1.5 * (self.__x - self.__y) * self.__length
        # self.__y_location = sqrt(3) / 2 * (self.__x + self.__y) * self.__length
        # if g.map_style == 'h':
        #     self.__x_location = 1.5 * (self.__x - self.__y) * self.__length
        #     self.__y_location = sqrt(3) / 2 * (self.__x + self.__y) * self.__length
        # else:
        #     self.__x_location = self.__length * self.__x
        #     self.__y_location = self.__length * self.__y

    @property
    def index(self):
        return [self.__x, self.__y]

    def clear(self):
        for sensor in self.__sensors:
            sensor.clear()

    def add_sensor(self, sensor):
        self.__sensors.append(sensor)

    def set_index(self, x, y):
        self.__x = x
        self.__y = y

    @property
    def position(self):
        return [self.__x_location, self.__y_location]

    def get_sensor_positions(self):
        x = []
        y = []
        for item in self.__sensors:
            [xt, yt] = item.getPosition()
            x.append(xt)
            y.append(yt)
        return x, y

    @property
    def sensors(self):
        return self.__sensors

    @property
    def sensor_number(self):
        return len(self.__sensors)

    def plot_cell(self, axis):
        x_val = [item + self.__x_location
                 for item in [self.__length * sqrt(3) / 2, 0, -self.__length * sqrt(3) / 2,
                              -self.__length * sqrt(3) / 2, 0, self.__length * sqrt(3) / 2,
                              self.__length * sqrt(3) / 2]]
        y_val = [item + self.__y_location
                 for item in [self.__length / 2, self.__length, self.__length / 2, -self.__length / 2,
                              -self.__length, -self.__length / 2, self.__length / 2]]
        axis.plot(x_val, y_val, color='k', linewidth=1)

    @staticmethod
    def plot_cells(cells: np.ndarray):
        [sensor_x, sensor_y] = Sensor.get_all_locations()
        map_fig = plt.figure(figsize=(10, 8), dpi=450)
        ax = map_fig.add_subplot(111)
        for row in cells:
            for cell in row:
                cell.plot_cell(ax)
        ax.scatter(sensor_x, sensor_y, color='r')
        plt.savefig('./cell_sensors.jpg')

    def uav_visited(self, current_slot):
        for sensor in self.__sensors:
            sensor.report_by_uav(current_slot)

    def worker_visited(self, current_slot):
        # 刷新状态，用于车辆访问的情况
        for sensor in self.__sensors:
            sensor.report_by_workers(current_slot)

    def get_observation_aoi(self, current_slot):
        ret = 0.0
        for sensor in self.__sensors:
            ret += sensor.get_observation_aoi(current_slot)
        return ret

    def get_real_aoi(self, current_slot):
        ret = 0.0
        for sensor in self.__sensors:
            ret += sensor.get_real_aoi(current_slot)
        return ret

    @staticmethod
    def uniform_generator_with_position(x_limit: int,
                                        y_limit: int,
                                        positions: np.ndarray,
                                        sensor_number: int,
                                        side_length: float,
                                        seed: object = 10) -> np.ndarray:

        ret_cell = np.empty(shape=(x_limit, y_limit), dtype=object)
        for x in range(x_limit):
            for y in range(y_limit):
                ret_cell[x, y] = Cell(x, y, positions[x][y], side_length)

        random.seed(seed)
        sensor_x = random.randint(0, x_limit, sensor_number)
        sensor_y = random.randint(0, y_limit, sensor_number)

        sensor_x_diff = random.uniform(-side_length * sqrt(3) / 2, side_length * sqrt(3) / 2, sensor_number)
        sensor_y_diff = np.array([random.uniform(abs(x_diff) / sqrt(3) - side_length,
                                                 - abs(x_diff) / sqrt(3) + side_length) for x_diff in sensor_x_diff])

        # print("sensors_x" + str(sensor_x[:20]) + 'sensor_y' + str(sensor_y[:20]))
        # print("x_diff" + str(sensor_x_diff[:20]) + 'y_diff' + str(sensor_y_diff[:20]))
        for i in range(sensor_number):
            Sensor(i, sensor_x_diff[i], sensor_y_diff[i], ret_cell[sensor_x[i], sensor_y[i]])

        return ret_cell

    # @staticmethod
    # def uniform_generator(cell_limit, cell_length, sensor_number, seed=10):
    #     random.seed(seed)
    #     ret_cell = []
    #     map_style = g.map_style
    #     for x in range(cell_limit):
    #         ret_cell.append([Cell(x, y) for y in range(cell_limit)])
    #     sensor_cell_x = [random.randint(0, cell_limit - 1) for _ in range(sensor_number)]
    #     sensor_cell_y = [random.randint(0, cell_limit - 1) for _ in range(sensor_number)]
    #
    #     r3 = sqrt(3)
    #     if map_style == 'g':
    #         sensor_x_diff = [random.uniform(0, cell_length) for _ in range(sensor_number)]
    #         sensor_y_diff = [random.uniform(0, cell_length) for _ in range(sensor_number)]
    #     elif map_style == 'h':
    #         sensor_x_diff = [random.uniform(-cell_length, cell_length) for _ in range(sensor_number)]
    #         sensor_y_diff = [random.uniform(-cell_length * r3 / 2, cell_length * r3 / 2) if abs(x) < cell_length / 2
    #                          else random.uniform(abs(x) * r3 - r3 * cell_length, - abs(x) * r3 + r3 * cell_length)
    #                          for x in sensor_x_diff]
    #     else:
    #         assert False
    #
    #     for i in range(sensor_number):
    #         cell_location_x, cell_location_y = ret_cell[sensor_cell_x[i]][sensor_cell_y[i]].position
    #         sen = Sensor(sensor_x_diff[i] + cell_location_x, sensor_y_diff[i] + cell_location_y,
    #                      sensor_cell_x[i], sensor_cell_y[i])
    #         ret_cell[sensor_cell_x[i]][sensor_cell_y[i]].add_sensor(sen)
    #     return ret_cell

# def NormalGenerator(xsize, ysize, cellsize=10, sensornum=500, seed = 10):
#     # 待修改
#     random.seed(seed)
#     retCell = []
#     for x in range(xsize):
#         retCell.append([Cell(x, y, cellsize) for y in range(ysize)])
#     c1 = [xsize * cellsize / 3, ysize * cellsize * 2 / 3]
#     c2 = [xsize * cellsize * 2 / 3, ysize * cellsize / 3]
#
#     senx = [random.normalvariate(0, 1) * cellsize for _ in range(sensornum)]
#     seny = [random.normalvariate(0, 1) * cellsize for _ in range(sensornum)]
#
#     for i in range(floor(sensornum / 2) ):
#         sen = Sensor(senx[i] + c1[0], seny[i] + c1[1], cellsize)
#         [cx, cy] = sen.getCellIndex()
#         retCell[cx][cy].addSensor(sen)
#
#     for i in range(floor(sensornum / 2) + 1, sensornum):
#         sen = Sensor(senx[i] + c2[0], seny[i] + c2[1], cellsize)
#         [cx, cy] = sen.getCellIndex()
#         retCell[cx][cy].addSensor(sen)
#
#     return retCell


if __name__ == '__main__':
    cleaner = DataCleaner()
    c = Cell.uniform_generator_with_position(cleaner.x_limit,
                                             cleaner.y_limit,
                                             cleaner.cell_coordinate,
                                             5000,
                                             cleaner.side_length)
    Cell.plot_cells(c)
