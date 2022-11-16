from global_parameter import Global as g
from math import sqrt


class Energy:
    __move_energy_cost = 0
    __hover_energy_cost = 0
    __charge_energy_gain = 0

    @staticmethod
    def init():
        v = g.uav_speed     # 无人机速度
        W = 1.2             # 重量
        D = 1.29            # 空气密度
        x = 0.221           # 长度
        y = 0.0963          # 宽度
        z = 0.0903          # 高度    https://www.dji.com/cn/mavic-3/specs
        A = 0.4             # 前迎面积
        Cd = 0.040          # 空气的动力阻力系数
        charge_power = 65   # 充电电压65w
        hover_energy_power = (W ** 2) / D / (x * y) / v
        move_energy_power = 0.5 * Cd * A * D * (v ** 3) + hover_energy_power
        second_span_cell = g.sec_per_slot

        Energy.__move_energy_cost = move_energy_power * second_span_cell / 3600
        Energy.__hover_energy_cost = hover_energy_power * second_span_cell / 3600
        Energy.__charge_energy_gain = charge_power * second_span_cell / 3600

    @staticmethod
    def hover_energy_cost():
        return Energy.__hover_energy_cost

    @staticmethod
    def move_energy_cost():
        return Energy.__move_energy_cost

    @staticmethod
    def charge_energy_one_slot():
        return Energy.__charge_energy_gain
