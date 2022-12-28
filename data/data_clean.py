# dataset clean
from math import sqrt, pi, ceil, inf
import os
import json
from zipfile import ZipFile
import numpy as np
from threading import Lock, Thread
import matplotlib.pyplot as plt
import math
from numpy import random

class DataCleaner:
    earth_radian_coefficient = pi * 6371393 / 180

    def __init__(self,
                 read_thread=8,
                 x_limit=10,
                 y_limit=10,
                 x_range=[116.15, 116.64],
                 y_range=[39.72, 40.095],
                 uav_speed=15,
                 sen_number=20000,
                 prop_1 = 3,    # dB
                 prop_2 = 23,   # dB
                 Pt = 21,       # dBm
                 N0 = -51,     # dBm
                 Height = 50,
                 Zeta = 11.95,
                 Pi = 0.14,
                 fc = 2e9,
                 band_width = 4 * 1024 * 1024,
                 msg_size = 1 * 1024 * 1024,
                 Ptrans = 0.0126):

        current_work_path = os.path.split(os.path.realpath(__file__))[0]
        data_set_json_path = current_work_path + '/data_set.json'

        sub_dir = current_work_path + '/data_x_' + str(x_limit) + '_y_' + str(y_limit)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        with open(data_set_json_path, 'r') as file:
            json_file = json.load(file)

        self.__x_limit = x_limit
        self.__y_limit = y_limit
        self.__x_range = x_range
        self.__y_range = y_range
        self.__uav_speed = uav_speed
        self.__sensor_number = sen_number

        self.__K0 = (4 * math.pi * fc / 299792458) ** (-Zeta)
        self.__prob_los = lambda d : 1 / (1 + Zeta * math.exp(-Pi * (180 / math.pi / math.sin(Height / math.sqrt(Height**2 + d**2)) - Zeta)))
        self.__prob_avg = lambda d : 1 / (10 ** prop_1) * self.__prob_los(d) + 1 / (10 ** prop_2) * (1 - self.__prob_los(d))
        self.__Ki = lambda d : self.__prob_avg(d) * self.__K0 * d ** (-Zeta)
        self.__SNR = Pt - N0 # dB
        self.__bit_rate = lambda d : band_width * math.log2(1 + 10 ** self.__SNR * self.__Ki(d))
        self.__energy_consume = lambda x, y : Ptrans * msg_size * 8 / self.__bit_rate(sqrt((x * DataCleaner.earth_radian_coefficient)**2 + (y * DataCleaner.earth_radian_coefficient)**2))
        self.__msg_size = msg_size
        self.__file_number = json_file['number']
        self.__prefix = json_file['prefix']
        self.__suffix = json_file['suffix']
        self.__pack_path = current_work_path + '/' + json_file['name']
        self.__unpack_directory = self.__pack_path[:self.__pack_path.find('.')] + '/'
        self.__data_directory = self.__unpack_directory + json_file['data_directory'] + '/'

        self.__coordinate_directory = current_work_path + "/coordinate"
        self.__coordinate_path = self.__coordinate_directory + '.npy'

        self.__info_json_path = sub_dir + '/' + json_file['name'][:json_file['name'].find('.')] + '.json'


        self.__coordinate = np.empty(shape=(0, 2))
        self.__worker_position = np.empty(shape=(self.__file_number,), dtype=dict)
        self.__cell_coordinate = np.empty(shape=(x_limit, y_limit), dtype=np.ndarray)

        self.__sensor_cell = np.empty(shape=(0, 2))
        self.__sensor_diff = np.empty(shape=(0, 2))

        self.__min_second = inf
        self.__max_second = 0
        self.__total_number = 0
        # 线程相关
        self.__lock = Lock()
        self.__thread_number = read_thread
        self.__read_number = ceil(self.__file_number / self.__thread_number)

        # self.__io_worker = ThreadPoolExecutor(max_workers=self.__thread_number)
        self.check_unpack()
        self.__info_json = self.read_info_json()
        self.__worker_position_directory = sub_dir + '/worker_sec_' + str(self.second_per_slot)
        self.__worker_position_path = self.__worker_position_directory + '.npy'

        self.__cell_coordinate_directory = sub_dir + '/cell_coordinate'
        self.__cell_coordinate_path = self.__cell_coordinate_directory + '.npy'

        self.__sensor_cell_directory = sub_dir + '/sensor_cell_coordinate_sen_' + str(sen_number)
        self.__sensor_cell_path = self.__sensor_cell_directory + '.npy'

        self.__sensor_diff_directory = sub_dir + '/sensor_diff_coordinate_sen_' + str(sen_number)
        self.__sensor_diff_path = self.__sensor_diff_directory + '.npy'

        self.__coordinate = self.read_coordinate() # 全加载进来对性能开销太大了，画图的时候再打开
        self.__cell_coordinate = self.read_cell_coordinate()
        self.__worker_position = self.read_worker_position()       # 读取已经存储好的info_json，如果没有，则创建，并导入信息
        self.__sensor_cell = self.read_sensor_cell()
        self.__sensor_diff = self.read_sensor_diff()

        # self.plot_scatters()

    def bit_rate(self, ground_distance: float):
        return self.__bit_rate(ground_distance)

    def energy_consume(self, xdiff:float, ydiff:float):
        return self.__energy_consume(xdiff, ydiff)

    def check_unpack(self):
        if not os.path.exists(self.__unpack_directory):
            os.makedirs(self.__unpack_directory)

        if len(os.listdir(self.__unpack_directory)) == 0:
            self.unpack()

    def unpack(self):
        if self.__pack_path[self.__pack_path.find('.'):] == ".zip":
            with ZipFile(self.__pack_path, 'r') as f:
                for file in f.namelist():
                    f.extract(file, self.__unpack_directory)

    # 为每个worker计算时隙、小区位置
    def read_worker_position(self):
        print("Begin read worker coordinate")
        if os.path.exists(self.__worker_position_path):
            print("Worker coordinate data found, load from {}".format(self.__worker_position_path))
            return np.load(self.__worker_position_path, allow_pickle=True)
        print("Not found worker coordinate file, begin generate from dataset")
        info_json = self.read_info_json()
        tasks = [Thread(target=self.task_read_worker_position, args=(i, info_json))
                 for i in range(self.__thread_number)]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()
        np.save(self.__worker_position_directory, self.__worker_position)
        print("Save worker coordinate at {}".format(self.__worker_position_path))
        return self.__worker_position

    def read_sensor_cell(self):
        print("Begin read sensor cell")
        if os.path.exists(self.__sensor_cell_path):
            print("Sensor cell data found, load from {}".format(self.__sensor_cell_path))
            return np.load(self.__sensor_cell_path, allow_pickle=True)
        print("Not found sensor cell file, begin generate from dataset")
        random.seed(10)
        sensor_x = random.randint(0, self.x_limit, self.sensor_number)
        sensor_y = random.randint(0, self.y_limit, self.sensor_number)

        ret = np.stack([sensor_x, sensor_y]).T
        np.save(self.__sensor_cell_directory, ret)
        print("Save sensor cell at {}".format(self.__sensor_cell_path))
        return ret

    def read_sensor_diff(self):
        print("Begin read sensor diff")
        if os.path.exists(self.__sensor_diff_path):
            print("Sensor diff data found, load from {}".format(self.__sensor_diff_path))
            return np.load(self.__sensor_diff_path, allow_pickle=True)
        print("Not found sensor diff file, begin generate from dataset")
        random.seed(10)
        sensor_x_diff = random.uniform(-self.side_length * sqrt(3) / 2, self.side_length * sqrt(3) / 2, self.sensor_number)
        sensor_y_diff = np.array([random.uniform(abs(x_diff) / sqrt(3) - self.side_length,
                                                 - abs(x_diff) / sqrt(3) + self.side_length) for x_diff in sensor_x_diff])
        ret = np.stack([sensor_x_diff, sensor_y_diff]).T
        np.save(self.__sensor_diff_directory, ret)
        print("Save sensor diff at {}".format(self.__sensor_diff_path))
        return ret

    def task_read_worker_position(self, ind, info_json):
        side_length = info_json['side_length']
        second_per_slot = info_json['second_per_slot']
        x_range = info_json['x_range']
        y_range = info_json['y_range']
        min_second = info_json['min_second']
        positions = []
        for i in range(ind * self.__read_number + 1, min((ind + 1) * self.__read_number, self.__file_number + 1)):
            slot = -1
            position_info = {}
            with open(self.file_name(i)) as f:
                line_list = f.readlines()
            for line in line_list:
                temp = line.strip().split(',')
                date_time = line.strip().split(',')[1]
                temp_slot = int((DataCleaner.datetime_to_second(date_time) - min_second) / second_per_slot)
                if temp_slot <= slot:
                    continue
                [x, y] = [float(temp[-2]), float(temp[-1])]
                if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1]:
                    continue
                slot = temp_slot
                cell_x, cell_y = self.nearest_cell(x, y)
                position_info[slot] = [cell_x, cell_y]

                # [bias_x, bias_y] = [x - x_range[0], y - y_range[0]]
                # remain_y = bias_y % (1.5 * side_length)
                # if 0.5 * side_length < remain_y < 1 * side_length:
                #     quot_y = int(bias_y / (1.5 * side_length))
                #     odd = quot_y % 2
                #     quot_x = int(bias_x / (sqrt(3) * side_length))
                #     rela_x = bias_x % (sqrt(3) * side_length)
                #     rela_y = (bias_y - side_length / 2) % (1.5 * side_length)
                #     left = (bias_x % (sqrt(3) * side_length) < (sqrt(3) / 2 * side_length))
                #     if odd and left:
                #         up = rela_y > rela_x / sqrt(3)
                #         cell_x = quot_x
                #         cell_y = quot_y + 1 if up else quot_y
                #     elif odd and not left:
                #         up = rela_y > (side_length - rela_x / sqrt(3))
                #         cell_x = quot_x + 1 if up else quot_x
                #         cell_y = quot_y + 1 if up else quot_y
                #     elif not odd and left:
                #         up = rela_y > (side_length / 2 - rela_x / sqrt(3))
                #         cell_x = quot_x
                #         cell_y = quot_y + 1 if up else quot_y
                #     else:
                #         up = rela_y > rela_x / sqrt(3)
                #         cell_x = quot_x if up else quot_x + 1
                #         cell_y = quot_y + 1 if up else quot_y
                # else:
                #     up = remain_y > 1 * side_length
                #     quot_y = int(bias_y / (1.5 * side_length))
                #     odd = quot_y % 2
                #     if up and odd:
                #         cell_y = quot_y
                #         cell_x = int((bias_x / (sqrt(3) * side_length) - 1 / 2))
                #     elif up and not odd:
                #         cell_y = quot_y + 1
                #         cell_x = int(bias_x / (sqrt(3) * side_length))
                #     elif not up and odd:
                #         cell_y = quot_y
                #         cell_x = int(bias_x / (sqrt(3) * side_length))
                #     else:
                #         cell_y = quot_y + 1
                #         cell_x = int((bias_x / (sqrt(3) * side_length) - 1 / 2))
                # position_info[slot] = [cell_x, cell_y]
            positions.append(position_info)
            self.__total_number += 1
            print("work {} processed, total {} processed\n".format(i, self.__total_number, ind), end="")
        self.__lock.acquire()
        for i in range(ind * self.__read_number + 1, min((ind + 1) * self.__read_number, self.__file_number + 1)):
            self.__worker_position[i - 1] = positions[i - (ind * self.__read_number + 1)]
        self.__lock.release()
        # print("thread {} end".format(ind))

    def plot_scatters(self):
        x = []
        y = []

        for i in range(self.__x_limit):
            for j in range(self.__y_limit):
              x.append(self.__cell_coordinate[i,j][0])
              y.append(self.__cell_coordinate[i,j][1])

        plt.figure(figsize=(10, 8), dpi=450)

        plt.scatter(self.worker_coordinate()[:, 0], self.worker_coordinate()[:, 1], c='gray', s=0.1)
        plt.scatter(x, y)
        x_range = self.__info_json['x_range']
        y_range = self.__info_json['y_range']
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.savefig('cell_fig.jpg')


    def nearest_cell(self, pos_x, pos_y):
        dis = inf
        idx = [-1, -1]

        # x_about = int((pos_x - self.__x_range[0]) / (self.side_length * sqrt(3)))
        # y_about = int((pos_y - self.__y_range[0]) / (self.side_length * 1.5))
        for i in range(self.__x_limit):
            for j in range(self.__y_limit):
                temp = sqrt((pos_x - self.__cell_coordinate[i, j][0]) ** 2 + (pos_y - self.__cell_coordinate[i, j][1]) ** 2)
                if temp < dis:
                    idx = [i, j]
                    dis = temp
        return idx

    def read_cell_coordinate(self):
        print("Begin read cell coordinate")
        if os.path.exists(self.__cell_coordinate_path):
            print("Cell coordinate data found, load from {}".format(self.__cell_coordinate_path))
            return np.load(self.__cell_coordinate_path, allow_pickle=True)
        print("Not found cell coordinate file, begin generate from dataset")
        side_length = self.__info_json['side_length']
        x_span = side_length * sqrt(3)
        y_span = side_length * 3 / 2
        for i in range(self.__y_limit):
            for j in range(self.__x_limit):
                dis = (x_span / 2 if i % 2 else 0)
                self.__cell_coordinate[i, j] = np.array([self.__x_range[0] + j * x_span + dis,
                                                         self.__y_range[0] + i * y_span])

        np.save(self.__cell_coordinate_directory, self.__cell_coordinate)
        print("Save cell coordinate at {}".format(self.__cell_coordinate_path))
        return self.__cell_coordinate

    # 读取已经处理好的json信息
    def read_info_json(self):
        if os.path.exists(self.__info_json_path):
            with open(self.__info_json_path, 'r') as file:
                info_json = json.load(file)
            if info_json['uav_speed'] != self.__uav_speed or info_json['x_limit'] != self.__x_limit \
                    or info_json['y_limit'] != self.__y_limit:
                if os.path.exists(self.__info_json_path):
                    os.remove(self.__info_json_path)
                if os.path.exists(self.__cell_coordinate_path):
                    os.remove(self.__cell_coordinate_path)
                if os.path.exists(self.__worker_position_path):
                    os.remove(self.__worker_position_path)
            else:
                return info_json
        # 缺失json文件，需要对数据进行预处理
        assert self.__x_limit > 1 and self.__y_limit > 1
        side_length = (self.__x_range[1] - self.__x_range[0]) / (self.__x_limit - 0.5) / sqrt(3)

        length_span_cell = side_length * DataCleaner.earth_radian_coefficient * sqrt(3)
        second_per_slot = max(length_span_cell / self.__uav_speed,
                              self.__msg_size * self.__sensor_number / (self.x_limit * self.y_limit)
                              / self.bit_rate(length_span_cell / sqrt(3)))
        x_span = side_length * sqrt(3)
        y_span = side_length * 3 / 2
        info_json = {
            'side_length': side_length,             # 六边形边长(弧度)
            'length_span_cell': length_span_cell,   # 跨小区移动距离(m)
            'second_per_slot': second_per_slot,
            'x_range': self.__x_range,
            'y_range': [self.__y_range[0], self.__y_range[0] + (self.__y_limit - 1) * y_span],
            'x_limit': self.__x_limit,
            'y_limit': self.__y_limit,
            'uav_speed': self.__uav_speed,
            'msg_size': self.__msg_size,
            'sensor_number': self.__sensor_number
        }

        # 处理时间信息，寻找车辆记录的起始时间和结束时间
        tasks = [Thread(target=self.task_time_analyze, args=(i,)) for i in range(self.__thread_number)]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()

        info_json['min_second'] = self.__min_second
        info_json['max_second'] = self.__max_second
        info_json['slot_number'] = int((info_json['max_second'] - info_json['min_second'] + 1)
                                       / second_per_slot)

        with open(self.__info_json_path, "w") as f:
            json.dump(info_json, f, indent=2)

        return info_json

    # 根据x数量，计算单个小区边长等信息
    def data_clean(self, x_range, y_range, x_number, y_number, uav_speed):
        pass

    def task_read_coordinate(self, ind):
        x = []
        y = []
        for i in range(ind * self.__read_number + 1, min((ind + 1) * self.__read_number, self.__file_number + 1)):
            with open(self.file_name(i)) as f:
                line_list = f.readlines()
                for line in line_list:
                    temp = line.strip().split(',')
                    x.append(float(temp[-2]))
                    y.append(float(temp[-1]))

        self.__lock.acquire()
        self.__coordinate = np.append(self.__coordinate, np.stack([x, y]).T, axis=0)
        self.__lock.release()
 
    def read_coordinate(self):
        print("Begin read dataset coordinate")
        if os.path.exists(self.__coordinate_path):
            print("Dataset coordinate data found, load from {}".format(self.__coordinate_path))
            return np.load(self.__coordinate_path, allow_pickle=True)
        print("Not found dataset coordinate file, begin generate from dataset")
        tasks = [Thread(target=self.task_read_coordinate, args=(i,)) for i in range(self.__thread_number)]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()

        np.save(self.__coordinate_directory, self.__coordinate)
        print("Save dataset coordinate at {}".format(self.__coordinate_path))
        return self.__coordinate

    @staticmethod
    def datetime_to_second(date_time):
        [date, time] = date_time.split(' ')
        [hour, minute, sec] = [int(t) for t in time.split(':')]
        [year, month, day] = [int(t) for t in date.split('-')]
        return sec + minute * 60 + hour * 3600 + day * 86400

    def task_time_analyze(self, ind):
        # print("time analysis id {} begin".format(ind))
        min_time = 999999999999999999
        max_time = 0
        for i in range(ind * self.__read_number + 1, min((ind + 1) * self.__read_number, self.__file_number + 1)):
            with open(self.file_name(i)) as f:
                line_list = f.readlines()
                if len(line_list) > 0:
                    for line in (line_list[0], line_list[-1]):
                        date_time = line.strip().split(',')[1]
                        sum_time = DataCleaner.datetime_to_second(date_time)
                        min_time = min(sum_time, min_time)
                        max_time = max(max_time, sum_time)

        self.__lock.acquire()
        self.__min_second = min(min_time, self.__min_second)
        self.__max_second = max(max_time, self.__max_second)
        self.__lock.release()

    def file_name(self, ind):
        return self.__data_directory + self.__prefix + str(ind) + self.__suffix

    @property
    def sensor_cell(self):
        return self.__sensor_cell

    @property
    def sensor_diff(self):
        return self.__sensor_diff

    def worker_coordinate(self):
        if self.__coordinate.shape[0] == 0:
            self.__coordinate = self.read_coordinate()
        return self.__coordinate

    @property
    def worker_position(self):
        return self.__worker_position

    @property
    def cell_coordinate(self):
        return self.__cell_coordinate

    @property
    def x_limit(self):
        return self.__x_limit

    @property
    def y_limit(self):
        return self.__y_limit

    @property
    def cell_limit(self):
        return [self.__x_limit, self.__y_limit]

    @property
    def sensor_number(self):
        return self.__info_json['sensor_number']
    @property
    def side_length(self):
        return self.__info_json['side_length']

    @property
    def second_per_slot(self):
        return self.__info_json['second_per_slot']

    @property
    def x_range(self):
        return self.__info_json['x_range']

    @property
    def y_range(self):
        return self.__info_json['y_range']

    @property
    def length_span_cell(self):
        return self.__info_json['length_span_cell']

    @property
    def slot_number(self):
        return self.__info_json['slot_number']

    @property
    def info_json(self):
        return self.__info_json

    @property
    def uav_speed(self):
        return self.__info_json['uav_speed']

    @property
    def msg_size(self):
        return self.__info_json['msg_size']


    
if __name__ == "__main__":
    cleaner = DataCleaner()
    # cleaner.plot_scatters()