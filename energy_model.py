from global_ import Global
from math import sqrt


class Energy:
    __move_energy_cost = 0
    __hover_energy_cost = 0

    @staticmethod
    def init():
        g = Global()
        v = g["uav_speed"]
        W = 1.2
        D = 1.29
        x = 0.221
        y = 0.0963
        z = 0.0903
        A = 0.4
        Cd = 0.040
        hover_energy_power = (W ** 2) / D / (x * y) / v
        move_energy_power = 0.5 * Cd * A * D * (v ** 3) + hover_energy_power

        if g["map_style"] == 'h':
            Energy.__move_energy_cost = move_energy_power * sqrt(3) * g["cell_length"] / v
            Energy.__hover_energy_cost = hover_energy_power * sqrt(3) * g["cell_length"] / v
        elif g["map_style"] == 'g':
            Energy.__move_energy_cost = move_energy_power * g["cell_length"] / v
            Energy.__hover_energy_cost = hover_energy_power * g["cell_length"] / v
        else:
            assert False

    @staticmethod
    def hover_energy_cost():
        return Energy.__hover_energy_cost

    @staticmethod
    def move_energy_cost():
        return Energy.__move_energy_cost
