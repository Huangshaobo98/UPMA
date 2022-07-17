from global_ import Global
from cell_model import uniform_generator
from energy_model import Energy
from environment_model import Environment

def global_init():
    g = {
        "map_style": 'h',                       # 地图类型h: 六边形 g: 栅格
        "cell_limit": 6,                              # 边界
        "cell_length": 48,                      # 小区边长
        "out_able": True,                      # 是否可以越出边界移动（仅对于worker）
        "worker_number": 20,
        "worker_initial_trust": 0.5,
        "worker_activity": 0.3,              # worker跨区移动的可能性
        "sensor_number": 500,
        "uav_speed": 12,                     # m/s
        "uav_energy": 20,                    # Wh
        "max_slot": 3000,
        "punish": 100,
        "batch_size": 256,
        "eposide": 2000,
        "charge_time": 60,
        "max_eposide": 1000,
        "initial_trust": 0.5,
        "hover_punish": 10
    }
    if g["cell_limit"] == 6:
        g["charge_cells"] = [[1, 1], [4, 4], [1, 4], [4, 1]]
    elif g["cell_limit"] == 10:
        g["charge_cells"] = [[2, 2], [7, 7], [2, 7], [7, 2]]

    return g


if __name__ == '__main__':
    Global.init(global_init())
    env = Environment()
    env.start()
