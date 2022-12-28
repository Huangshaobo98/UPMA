from global_parameter import Global as g
from data.data_clean import DataCleaner

class Energy:
    # v = g.uav_speed  # 无人机速度
    W = 1.2  # 重量
    D = 1.29  # 空气密度
    x = 0.3475  # 长度
    y = 0.283  # 宽度
    z = 0.1077  # 高度    https://www.dji.com/cn/mavic-3/specs
    A = 0.0406707  # 前迎面积
    Cd = 0.0895  # 空气的动力阻力系数
    charge_power = 65  # 充电电压65w
    __move_energy_cost = -1
    __hover_energy_cost = -1
    __charge_energy_gain = -1

    @staticmethod
    def init(cleaner: DataCleaner):
        hover_energy_power = (Energy.W ** 2) / Energy.D / (Energy.x * Energy.y) / cleaner.uav_speed
        move_energy_power = 0.5 * Energy.Cd * Energy.A * Energy.D * (cleaner.uav_speed ** 3) + hover_energy_power
        Energy.__move_energy_cost = move_energy_power * cleaner.second_per_slot / 3600
        Energy.__hover_energy_cost = hover_energy_power * cleaner.second_per_slot / 3600
        Energy.__charge_energy_gain = Energy.charge_power * cleaner.second_per_slot / 3600

    @staticmethod
    def hover_energy_cost():
        return Energy.__hover_energy_cost

    @staticmethod
    def move_energy_cost():
        return Energy.__move_energy_cost

    @staticmethod
    def charge_energy_one_slot():
        return Energy.__charge_energy_gain
