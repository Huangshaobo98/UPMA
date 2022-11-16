from math import sqrt


class Global:
    # 网络模型相关
    map_style = 'h'             # 地图类型h: 六边形 g: 栅格
    cell_limit = 6              # 边界大小
    cell_length = 60            # 小区边长 m
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
    hover_punish = 50           # 悬停惩罚
    charge_everywhere = True    # 是否可以任意位置充电

    # 训练相关
    max_slot = 5000             # 最大时隙
    punish = 200                # 惩罚因子？
    batch_size = 256            # 批大小
    # charge_time = 60          # 充电耗时(废弃参数)
    max_episode = 1000          # 最大训练episode

    # 充电位置
    charge_cells = [[1, 1], [4, 4], [1, 4], [4, 1]]     # 待设置的参数

    # 其他参数(基于上述参数计算得到的)，在init中进行初始化
    sec_per_slot = 0

    # init方法只用来检测是否正确的初始化了，可能后续有一些参数需要进行验证?
    @staticmethod
    def init():
        Global.check()
        sec_per_slot = Global.cell_length / Global.uav_speed
        if Global.map_style == 'h':
            sec_per_slot *= sqrt(3)

    @staticmethod
    def check():
        assert Global.map_style == 'h' or Global.map_style == 'g'
        assert Global.cell_limit is int
        assert Global.worker_number is int
        assert Global.max_slot is int
        assert Global.max_episode is int
        assert Global.batch_size is int
