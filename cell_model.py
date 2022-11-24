from sensor_model import Sensor
import random
from math import sqrt
from global_parameter import Global as g


class Cell:
    def __init__(self, x=0, y=0):
        self.__x = x
        self.__y = y
        self.__length = g.cell_length
        self.__map_style = g.map_style
        if g.map_style == 'h':
            self.__x_location = 1.5 * (self.__x - self.__y) * self.__length
            self.__y_location = sqrt(3) / 2 * (self.__x + self.__y) * self.__length
        else:
            self.__x_location = self.__length * self.__x
            self.__y_location = self.__length * self.__y
        self.__sensors = []

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
        return self.__x_location, self.__y_location

    def get_sensor_positions(self):
        x = []
        y = []
        for item in self.__sensors:
            [xt, yt] = item.getPosition()
            x.append(xt)
            y.append(yt)
        return x, y

    def get_sensors(self):
        return self.__sensors

    def plot_cell(self, axis):
        if self.__map_style == 'h':
            x_val = [item + self.__x_location for item in [-self.__length, -self.__length / 2,
                     self.__length / 2, self.__length, self.__length / 2, -self.__length / 2, -self.__length]]
            y_val = [item + self.__y_location for item in [0, -self.__length * sqrt(3) / 2, -self.__length * sqrt(3) / 2, 0,
                     self.__length * sqrt(3) / 2, self.__length * sqrt(3) / 2, 0]]
        elif self.__map_style == 'g':
            x_val = [item + self.__x_location for item in [-self.__length / 2, self.__length / 2,
                     self.__length / 2, -self.__length / 2]]
            y_val = [item + self.__y_location for item in [-self.__length / 2, -self.__length / 2,
                     self.__length / 2, self.__length / 2]]
        else:
            assert False

        axis.plot(x_val, y_val, color='k', linewidth=1)

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


def uniform_generator(seed=10):
    random.seed(seed)
    ret_cell = []
    cell_limit = g.cell_limit
    cell_length = g.cell_length
    sensor_number = g.sensor_number
    map_style = g.map_style
    for x in range(cell_limit):
        ret_cell.append([Cell(x, y) for y in range(cell_limit)])
    sensor_cell_x = [random.randint(0, cell_limit - 1) for _ in range(sensor_number)]
    sensor_cell_y = [random.randint(0, cell_limit - 1) for _ in range(sensor_number)]

    r3 = sqrt(3)
    if map_style == 'g':
        sensor_x_diff = [random.uniform(0, cell_length) for _ in range(sensor_number)]
        sensor_y_diff = [random.uniform(0, cell_length) for _ in range(sensor_number)]
    elif map_style == 'h':
        sensor_x_diff = [random.uniform(-cell_length, cell_length) for _ in range(sensor_number)]
        sensor_y_diff = [random.uniform(-cell_length * r3 / 2, cell_length * r3 / 2) if abs(x) < cell_length / 2
                         else random.uniform(abs(x) * r3 - r3 * cell_length, - abs(x) * r3 + r3 * cell_length)
                         for x in sensor_x_diff]
    else:
        assert False

    for i in range(sensor_number):
        cell_location_x, cell_location_y = ret_cell[sensor_cell_x[i]][sensor_cell_y[i]].position
        sen = Sensor(sensor_x_diff[i] + cell_location_x, sensor_y_diff[i] + cell_location_y,
                     sensor_cell_x[i], sensor_cell_y[i])
        ret_cell[sensor_cell_x[i]][sensor_cell_y[i]].add_sensor(sen)
    return ret_cell

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
