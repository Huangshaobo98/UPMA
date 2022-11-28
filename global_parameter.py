from math import sqrt, log, exp
import os


class Global:
    # 网络模型相关
    map_style = 'h'             # 地图类型h: 六边形 g: 栅格
    cell_limit = 6              # 边界大小
    cell_length = 600           # 小区边长 m
    out_able = True             # 可移动离开所观测小区

    # worker相关
    worker_number = 2
    worker_initial_trust = 0.5
    worker_activity = 0.4       # 活跃度，每轮以此概率移动
    initial_trust = 0.5
    worker_work_rate = 0.6      # worker在每个时隙下工作的概率
    worker_start_fix = False

    # 传感器相关
    sensor_number = 500

    # 无人机相关
    uav_speed = 15              # m/s
    uav_energy = 20             # Wh
    uav_start_fix = True        # 无人机初始位置是否固定
    uav_start_location = [0, 0]
    hover_punish = 1            # 悬停惩罚
    charge_everywhere = True    # 是否可以任意位置充电

    # 训练相关
    max_slot = 3000            # 最大时隙
    no_power_punish = cell_limit * cell_limit      # 无电量惩罚
    batch_size = 256            # 批大小
    # charge_time = 60          # 充电耗时(废弃参数)
    max_episode = 1000          # 最大训练episode
    onehot_position = True      # 位置编码为onehot形式
    energy_reward_point = [0.1, -0.1]    # 能量低于point[0]时，reward下降point[1]
    energy_reward_coefficient = -log(energy_reward_point[0] / (2 + energy_reward_point[1])) \
                                / energy_reward_point[1]

    # 充电位置
    charge_cells = [[1, 1], [4, 4], [1, 4], [4, 1]]     # 待设置的参数

    # 其他参数(基于上述参数计算得到的)，在init中进行初始化
    sec_per_slot = cell_length / uav_speed * (sqrt(3) if map_style == 'h' else 1)

    # 存储路径参数
    default_save_path = os.getcwd() + "/save/cell_" + str(cell_limit)

    # 持久化参数
    default_train = True
    default_continue_train = False

    # 日志配置参数
    default_file_log = False
    default_console_log = True

    # 数据分析模式，开启此状态将读取训练/测试数据进行数据分析，绘图等工作
    default_analysis = False

    # init方法只用来检测是否正确的初始化了，可能后续有一些参数需要进行验证?
    @staticmethod
    def init():
        Global.check()

    @staticmethod
    def check():
        assert Global.map_style == 'h' or Global.map_style == 'g'
        assert type(Global.cell_limit) is int
        assert type(Global.worker_number) is int
        assert type(Global.max_slot) is int
        assert type(Global.max_episode) is int
        assert type(Global.batch_size) is int

    @staticmethod
    def energy_reward_calculate(x):
        return - 2 * exp(Global.energy_reward_coefficient * x) \
                                        / (1 + exp(Global.energy_reward_coefficient * x))