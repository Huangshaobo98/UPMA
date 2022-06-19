from global_ import Global
from plot_model import plot_figure
from cell_model import uniform_generator
from energy_model import Energy


def global_init():
    g = {
        "map_style": 'h',                       # 地图类型h: 六边形 g: 栅格
        "x_limit": 10,                              # 边界
        "y_limit": 10,
        "out_able": True,                      # 是否可以越出边界移动（仅对于worker）
        "worker_number": 20,
        "worker_initial_trust": 0.5,
        "worker_activity": 0.3,              # worker跨区移动的可能性
        "cell_length": 10,                      # 小区边长
        "sensor_number": 500,
        "uav_speed": 12,                     # m/s
        "uav_energy": 20                    # Wh
    }
    return g


if __name__ == '__main__':
    Global.init(global_init())
    Energy.init()
    cells = uniform_generator()
    plot_figure(cells)
