
from cell_model import Cell
from typing import List
from sensor_model import Sensor
import matplotlib.pyplot as plt


def plot_figure(cells: List[List[Cell]]):

    [sensor_x, sensor_y] = Sensor.get_all_locations()
    map_fig = plt.figure()
    ax = map_fig.add_subplot(111)
    for row in cells:
        for cell in row:
            cell.plot_cell(ax)
    ax.scatter(sensor_x, sensor_y, color='r')
    plt.show()

def plot_curve():
