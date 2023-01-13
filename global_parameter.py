from math import log, exp

# 用于存储一些非命令行调整的参数
# 前缀加default表示默认参数，实际参数可能不与默认参数一致


class Global:
    # 网络模型相关
    # map_style = 'h'             # 地图类型h: 六边形 g: 栅格
    # cell_length = 600           # 小区边长 m

    # worker相关
    worker_out_able = True      # 可移动离开所观测小区
    worker_initial_trust = 0.5
    worker_vitality = 5       # 活跃度，最大可收集的节点数量
    initial_trust = 0.5
    worker_work_rate = 0.6      # worker在每个时隙下工作的概率
    worker_start_fix = False

    # 无人机相关
    # uav_speed = 15              # m/s
    uav_energy = 77             # Wh
    uav_start_fix = True        # 无人机初始位置是否固定
    uav_start_location = [0, 0]
    charge_everywhere = True    # 是否可以任意位置充电

    # 训练相关
    # charge_time = 60          # 充电耗时(废弃参数)

    onehot_position = True      # 位置编码为onehot形式
    energy_reward_point = [0.1, -0.1]    # 能量低于point[0]时，reward下降point[1]
    energy_reward_coefficient = -log(energy_reward_point[0] / (2 + energy_reward_point[1])) / energy_reward_point[1]

    # 充电位置
    charge_cells = [[1, 1], [4, 4], [1, 4], [4, 1]]     # 待设置的参数

    # 下列参数为可以通过命令行自由调整的参数
    # default_cell_limit = 6          # 边界大小
    default_sensor_number = 5000     # 传感器数量
    default_worker_number = 500       # worker数量
    default_max_episode = 500      # 最大训练episode
    # default_max_slot = 3000         # 最大时隙
    default_batch_size = 256        # 批大小
    default_learn_rate = 0.0001     # 学习率
    default_gamma = 0.9                # 折扣系数
    default_epsilon_decay = 0.99999     # 探索率衰减
    default_detail_log = False
    default_compare = False
    default_malicious = 0.15
    default_assignment_aoi_reduce_rate = 0.875
    default_test_episode = 250
    default_windows_length = 10
    default_pho = 0.7
    default_random_assignment = False
    # 持久化参数
    default_train = True
    default_continue_train = False

    # 日志配置参数
    default_file_log = False
    default_console_log = False

    # 数据分析模式，开启此状态将读取训练/测试数据进行数据分析，绘图等工作
    default_analysis = False

    default_compare_method = ""
    @staticmethod
    def energy_reward_calculate(x):
        return - 2 * exp(Global.energy_reward_coefficient * x) \
                                        / (1 + exp(Global.energy_reward_coefficient * x))
