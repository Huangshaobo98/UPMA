# dataset clean
from math import sqrt, pi, ceil, inf
import os
import json
from zipfile import ZipFile
import numpy as np
from threading import Lock, Thread
import matplotlib.pyplot as plt


class DataCleaner:
    earth_radian_coefficient = pi * 6371393 / 180

    def __init__(self,
                 read_thread=8,
                 x_limit=16,
                 y_limit=16,
                 x_range=[116.15, 116.64],
                 y_range=[39.72, 40.095],
                 uav_speed=15):

        current_work_path = os.path.split(os.path.realpath(__file__))[0]
        data_set_json_path = current_work_path + '/data_set.json'

        with open(data_set_json_path, 'r') as file:
            json_file = json.load(file)

        self.__x_limit = x_limit
        self.__y_limit = y_limit
        self.__x_range = x_range
        self.__y_range = y_range
        self.__uav_speed = uav_speed

        self.__file_number = json_file['number']
        self.__prefix = json_file['prefix']
        self.__suffix = json_file['suffix']
        self.__pack_path = current_work_path + '/' + json_file['name']
        self.__unpack_directory = self.__pack_path[:self.__pack_path.find('.')] + '/'
        self.__data_directory = self.__unpack_directory + json_file['data_directory'] + '/'
        self.__coordinate_directory = current_work_path + "/coordinate"
        self.__coordinate_path = self.__coordinate_directory + '.npy'

        self.__info_json_path = current_work_path + '/' + json_file['name'][:json_file['name'].find('.')] + '.json'
        self.__worker_position_directory = current_work_path + '/worker'
        self.__worker_position_path = self.__worker_position_directory + '.npy'

        self.__cell_coordinate_directory = current_work_path + '/cell_coordinate'
        self.__cell_coordinate_path = self.__cell_coordinate_directory + '.npy'

        self.__coordinate = np.empty(shape=(0, 2))
        self.__worker_position = np.empty(shape=(self.__file_number,), dtype=dict)
        self.__cell_coordinate = np.empty(shape=(x_limit, y_limit), dtype=np.ndarray)
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
        self.__coordinate = self.read_coordinate()
        self.__cell_coordinate = self.read_cell_coordinate()
        self.__worker_position = self.read_worker_position()       # 读取已经存储好的info_json，如果没有，则创建，并导入信息
        self.plot_scatters()

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
        if os.path.exists(self.__worker_position_path):
            return np.load(self.__worker_position_path, allow_pickle=True)
        info_json = self.read_info_json()
        tasks = [Thread(target=self.task_read_worker_position, args=(i, info_json))
                 for i in range(self.__thread_number)]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()
        np.save(self.__worker_position_directory, self.__worker_position)
        print("worker saved")
        return self.__worker_position

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
            print("work {} over, total {}, id {}".format(i, self.__total_number, ind))
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

        plt.scatter(self.__coordinate[:, 0], self.__coordinate[:, 1], c='gray', s=0.1)
        plt.scatter(x, y)
        x_range = self.__info_json['x_range']
        y_range = self.__info_json['y_range']
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.savefig('./cell_fig.jpg')


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
        if os.path.exists(self.__cell_coordinate_path):
            return np.load(self.__cell_coordinate_path, allow_pickle=True)
        side_length = self.__info_json['side_length']
        x_span = side_length * sqrt(3)
        y_span = side_length * 3 / 2
        for i in range(self.__y_limit):
            for j in range(self.__x_limit):
                dis = (x_span / 2 if i % 2 else 0)
                self.__cell_coordinate[i, j] = np.array([self.__x_range[0] + j * x_span + dis,
                                                         self.__y_range[0] + i * y_span])

        np.save(self.__cell_coordinate_directory, self.__cell_coordinate)
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
        second_per_slot = length_span_cell / self.__uav_speed
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
            'uav_speed': self.__uav_speed
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
        if os.path.exists(self.__coordinate_path):
            return np.load(self.__coordinate_path, allow_pickle=True)
        tasks = [Thread(target=self.task_read_coordinate, args=(i,)) for i in range(self.__thread_number)]
        for task in tasks:
            task.start()
        for task in tasks:
            task.join()

        np.save(self.__coordinate_directory, self.__coordinate)

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
    def worker_coordinate(self):
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
    
if __name__ == "__main__":
    cleaner = DataCleaner()
