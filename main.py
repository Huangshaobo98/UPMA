from global_ import Global
from plot_model import plot_figure
from cell_model import uniform_generator
from energy_model import Energy


def global_init():
    g = Global()
    g["map_style"] = 'h'  # g for grid /  for hexagon
    g["x_limit"] = 10
    g["y_limit"] = 10
    g["out_able"] = True
    g["worker_initial_trust"] = 0.5
    g["worker_activity"] = 0.3
    g["cell_length"] = 10
    g["sensor_number"] = 500
    g["uav_speed"] = 12
    g["uav_energy"] = 77  # Wh


if __name__ == '__main__':
    global_init()
    Energy.init()
    cells = uniform_generator()
    plot_figure(cells)
