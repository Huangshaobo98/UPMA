
class Global:
    ## 网络模型相关
    map_style = 'h'             # 地图类型h: 六边形 g: 栅格
    cell_limit = 6              # 边界大小
    cell_length = 48            # 小区边长 m
    out_able = True             # 可移动离开所观测小区

    ## worker相关
    worker_number = 2
    worker_initial_trust = 0.5
    worker_activity = 0.4       # 活跃度，每轮以此概率移动
    initial_trust = 0.5

    ## 传感器相关
    sensor_number = 500

    ## 无人机相关
    uav_speed = 12              # m/s
    uav_energy = 20             # Wh
    uav_start_fix = False       # 无人机初始位置是否固定
    uav_start_location = [0, 0]
    hover_punish = 50           # 悬停惩罚
    charge_everywhere = True    # 是否可以任意位置充电

    ## 训练相关
    max_slot = 5000             # 最大时隙
    punish = 200                # 惩罚因子？
    batch_size = 256            # 批大小
    charge_time = 60            # 充电耗时
    max_eposide = 1000          # 最大训练eposide

    ## 充电位置
    charge_cells = [[1, 1], [4, 4], [1, 4], [4, 1]]     # 待设置的参数