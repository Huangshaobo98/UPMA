import os.path

import matplotlib.pyplot as plt
import pandas as pd

from data.data_clean import DataCleaner
from cell_model import Cell
from sensor_model import Sensor
from math import sqrt, log, log2, log10
import numpy as np
from analysis import *
import seaborn as sns

from matplotlib import rcParams
from matplotlib import font_manager
import scipy.io as sio
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font_path = "/home/huangshaobo/TimesHei.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
# print(prop.get_name())  # 显示当前使用字体的名称

# 字体设置
rcParams['font.family'] = 'sans-serif'  # 使用字体中的无衬线体
rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
rcParams['font.size'] = 18  # 设置字体大小
rcParams['axes.unicode_minus'] = False  # 使坐标轴刻度标签正常显示正负号

save_path = 'figure/'
root_path = '/home/huangshaobo/workspace/UPMA_random_worker_1/'


def plot_1_4_curve_fig(x_tuple, y_tuple,
                       x_label, y_label,
                       x_range_tuple, y_range_tuple,
                       title_tuple, legend_tuple, save_name):
    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    for idx, ax in enumerate(axs):
        for idx1 in range(4):
            ax.plot(x_tuple[idx][idx1], y_tuple[idx][idx1], color=plt.get_cmap('tab10')(idx1), label=legend_tuple[idx1])

        ax.set_xlim(x_range_tuple[idx])
        ax.set_ylim(y_range_tuple[idx])
        ax.set_xlabel(x_label[idx])
        ax.set_ylabel(y_label[idx])
        ax.set_title(title_tuple[idx], y=-0.26)
        ax.legend()
        ax.grid(linestyle='--')

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".tif")
    plt.savefig(save_path + save_name + ".jpg")


def plot_learn_and_gamma():
    save_name = save_path + 'new_learn_rate_gamma'
    [lr_data, lr_unfinish, gamma_data, gamma_unfinish] = learn_rate_analysis()
    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\mu=0.001$', '$\mu=0.0005$', '$\mu=0.0001$', '$\mu=0.00005$', '$\mu=0.00001$']
    for idx, i in enumerate(names):
        axl.plot([i for i in range(1, 501)], lr_data[idx], color=plt.get_cmap('tab10')(idx), label=i)
        print(idx)
    axl.set_xlabel('Episode')
    axl.set_ylabel('Average global AoI (${\\Delta t}$)')
    axl.set_xlim(0, 500)
    axl.set_ylim(10, 90)
    axl.legend()
    axl.set_title('(a) Learning rate $\mu$', y=-0.26)
    axl.grid(linestyle='--')
    axg = fig.add_subplot(122)

    names = ['$\gamma=0.99$', '$\gamma=0.95$', '$\gamma=0.9$', '$\gamma=0.75$', '$\gamma=0.5$']
    for idx, i in enumerate(names):
        axg.plot([i for i in range(1, 501)], gamma_data[idx], color=plt.get_cmap('Set1')(idx), label=i)
        print(idx)
    axg.set_xlabel('Episode')
    axg.set_ylabel('Average global AoI (${\\Delta t}$)')
    axg.set_xlim(0, 500)
    axg.set_ylim(10, 90)
    axg.legend()
    axg.set_title('(b) Discount rate $\gamma$', y=-0.26)
    axg.grid(linestyle='--')
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_name + '.tif')
    plt.savefig(save_name + '.jpg')


def plot_aoi_vary_with_worker_number():
    x_tuple, y_tuple = aoi_vary_with_worker_number()
    x_label = ["Slot" for _ in range(4)]
    y_label = ["Global AoI" for _ in range(4)]
    x_range = [[0, 1395] for _ in range(4)]
    y_range = [[0, 100], [0, 100], [0, 25], [0, 25]]
    lengd = ["DRL-GAM", "Greedy", "Robin Round", "CCPP"]
    titles = ["(a) M = 50", "(b) M = 500", "(c) M = 5000", "(d) M = 10000"]
    plot_1_4_curve_fig(x_tuple, y_tuple,
                       tuple(x_label), tuple(y_label),
                       tuple(x_range), tuple(y_range),
                       tuple(titles), tuple(lengd),
                       "aoi_under_workers")
    print("ok")


def plot_aoi_vary_with_sensor_number():
    x_tuple, y_tuple = aoi_vary_with_sensor_number()
    x_label = ["Slot" for _ in range(4)]
    y_label = ["Global AoI" for _ in range(4)]
    x_range = [[0, 1395] for _ in range(4)]
    y_range = [[0, 120], [0, 100], [0, 100], [0, 100]]
    lengd = ["DRL-GAM", "Greedy", "Robin Round", 'CCPP']
    titles = ["(a) N = 500", "(b) N = 1000", "(c) N = 5000", "(d) N = 10000"]
    plot_1_4_curve_fig(x_tuple, y_tuple,
                       tuple(x_label), tuple(y_label),
                       tuple(x_range), tuple(y_range),
                       tuple(titles), tuple(lengd),
                       "aoi_under_sensors")
    print("ok")


def plot_heat_map_aoi():
    dfs = global_average_aoi()
    names = ['DRL-GAM', 'Greedy', 'Robin Round']
    titles = ['(a) DRL-GAM', '(b) Greedy', '(c) Robin Round', '(d) CCPP']
    fig = plt.figure(figsize=(18, 4), dpi=300)
    axs = []
    for idx, i in enumerate(range(141, 145)):
        ax = fig.add_subplot(i)
        axs.append(ax)
        sns.heatmap(data=dfs[idx], ax=ax, annot=True, cmap="GnBu", vmax=1.5, vmin=0, fmt='.2f',
                    annot_kws={'fontsize': 4})
        ax.set_xlabel('Cell x')
        ax.set_ylabel('Cell y')
        ax.set_title(titles[idx], y=-0.26)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + 'heat_aoi.tif')
    plt.savefig(save_path + 'heat_aoi.jpg')


def plot_heat_map_visit_time():
    dfs = global_path_time()
    titles = ['(a) DRL-GAM', '(b) Greedy', '(c) Robin Round', '(d) CCPP']
    fig = plt.figure(figsize=(18, 4), dpi=300)
    axs = []
    for idx, i in enumerate(range(141, 145)):
        ax = fig.add_subplot(i)
        axs.append(ax)
        sns.heatmap(data=dfs[idx], ax=ax, annot=True, cmap="GnBu", vmax=0, vmin=32,
                    annot_kws={'fontsize': 6})
        ax.set_xlabel('Cell x')
        ax.set_ylabel('Cell y')
        ax.set_title(titles[idx], y=-0.26)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + 'heat_path_time.tif')
    plt.savefig(save_path + 'heat_path_time.jpg')


def plot_mean_ptp_std():
    datas = [mean, ptp, std] = mean_ptp_std()
    titles = ['(a) Mean value', '(b) Range', '(c) Standard deviation']
    fig = plt.figure(figsize=(14, 4), dpi=300)
    axs = []
    for idx, i in enumerate(range(131, 134)):
        ax = fig.add_subplot(i)
        axs.append(ax)
        sns.heatmap(data=datas[idx], ax=ax, annot=True, cmap="winter_r", cbar=False, fmt='.4f',
                    annot_kws={'fontsize': 12})
        ax.set_title(titles[idx], y=-0.26)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + 'heat_mean_ptp_std.tif')
    plt.savefig(save_path + 'heat_mean_ptp_std.jpg')


def pho_win_plot():
    save_name = save_path + 'pho_window'
    pho, good, bad = pho_settings()
    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\\rho=0.3$', '$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']

    for idx, i in enumerate(names):
        axl.plot([i for i in range(1, 1395)], pho[idx], color=plt.get_cmap('tab10')(idx), label=i)
        print(idx)
    axl.set_xlabel('Slots')
    axl.set_ylabel('2-Norm error of estimated and actual AoI')
    axl.set_xlim(0, 1395)
    # axl.set_ylim(0, 300)
    axl.legend()
    axl.set_title('(a) Effect of $\\rho$ on AoI estimation', y=-0.26)
    axl.grid(linestyle='--')

    axg = fig.add_subplot(122)

    names = ['$\\rho=0.3$', '$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']
    for idx, i in enumerate(names):
        axg.plot([i for i in range(1, 1395)], good[idx], color=plt.get_cmap('Set1')(idx), label='Normal: ' + i)
        axg.plot([i for i in range(1, 1395)], bad[idx], color=plt.get_cmap('Set1')(idx), linestyle='--',
                 label='Malicious: ' + i)
        print(idx)
    axg.set_xlabel('Slots')
    axg.set_ylabel('Average trust')
    axg.set_xlim(0, 1395)
    # axg.set_ylim(0, 300)
    axg.legend()
    axg.set_title('(b) Trust of malicious and normal workers', y=-0.26)
    axg.grid(linestyle='--')
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_name + '.tif')
    plt.savefig(save_name + '.jpg')


def mali_plot():
    save_name = save_path + 'mali_est_assign'
    [random_assign, random_norm, test_assign, test_norm] = mali_compare()
    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['malicious: 10%', 'malicious: 30%', 'malicious: 50%', 'malicious: 70%']
    # axl.grid(linestyle='--')
    axl.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) - 0.02, random_norm, width=0.04,
            color=plt.get_cmap('tab10')(0), label='Random')
    axl.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) + 0.02, test_norm, width=0.04, color=plt.get_cmap('tab10')(1),
            label='GMTA')

    axl.set_xlabel('Malicious ratio')
    axl.set_ylabel('2-Norm error of estimated and actual AoI')
    # axl.set_xlim(0.05)
    # axl.set_ylim(0, 300)
    axl.legend()
    axl.set_title('(a) Error of estimated and actual AoI', y=-0.26)
    axl.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    axg = fig.add_subplot(122)

    # names = ['$\\rho=0.3$', '$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']
    # axg.grid(linestyle='--')
    axg.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) - 0.02, random_assign, width=0.04,
            color=plt.get_cmap('tab10')(0), label='Random')
    axg.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) + 0.02, test_assign, width=0.04,
            color=plt.get_cmap('tab10')(1), label='GMTA')
    axg.set_xlabel('Malicious ratio')
    axg.set_ylabel('Task assignment ratio')
    # axg.set_xlim(0, 1395)
    # axg.set_ylim(0, 300)
    axg.legend()
    axg.set_title('(b) Ratio of tasks assigned to malicious workers', y=-0.26)

    axg.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_name + '.tif')
    plt.savefig(save_name + '.jpg')


def new_get_aoi_data_four_compare():
    workers = [100, 200, 500, 1000, 2000, 5000]
    sensors = [500, 1000, 5000, 10000]
    labels = ['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP']
    titles = ['(a) SN number $N$ = 500', '(b) SN number $N$ = 1000', '(c) SN number $N$ = 5000',
              '(d) SN number $N$ = 10000']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(sensors), len(workers)), dtype=np.float64)
    compare_reduce_rate = []
    for i in range(4):
        for idw, w in enumerate(workers):
            for ids, s in enumerate(sensors):
                pathname = 'save/t-drive_sen_{}_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999' \
                               .format(str(s), str(w)) + paths[i]
                npz = np.load(pathname)
                matrix[i, ids, idw] = np.average(npz['avg_real_aoi'][npz['actual_slot'] == 1394])
                if i > 0:
                    compare_reduce_rate.append((matrix[i, ids, idw] - matrix[0, ids, idw]) / matrix[i, ids, idw])
    minv = min(compare_reduce_rate)
    maxv = max(compare_reduce_rate)
    averagev = np.average(compare_reduce_rate)

    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    diff = [-0.3, -0.1, 0.1, 0.3]
    width = 0.2
    for idx, ax in enumerate(axs):

        for idx1 in range(4):
            ax.bar([i + diff[idx1] for i in range(len(workers))], matrix[idx1][idx, :], width=width,
                   color=plt.get_cmap('tab20')(idx1), label=labels[idx1], zorder=10)

        # ax.set_xlim(-1, len(workers))
        ax.set_xticks([i for i in range(len(workers))])
        ax.set_xticklabels(workers)

        ax.yaxis.grid(linestyle='--', zorder=0)
        ax.set_ylim(0, 50)
        ax.set_xlabel('Worker number $M$')
        ax.set_ylabel('Average global AoI (${\\Delta t}$)')
        ax.set_title(titles[idx], y=-0.26)
        ax.legend()
        # ax.grid(linestyle='--')

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + "new_aoi_sen_work.tif")
    plt.savefig(save_path + "new_aoi_sen_work.jpg")


def new_get_std_data_four_compare():
    workers = [100, 200, 500, 1000, 2000, 5000]
    sensors = [500, 1000, 5000, 10000]
    labels = ['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP']
    titles = ['(a) SN number $N$ = 500', '(b) SN number $N$ = 1000', '(c) SN number $N$ = 5000',
              '(d) SN number $N$ = 10000']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(sensors), len(workers)), dtype=np.float64)
    compare_reduce_rate = []
    for i in range(4):
        for idw, w in enumerate(workers):
            for ids, s in enumerate(sensors):
                pathname = 'save/t-drive_sen_{}_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999' \
                               .format(str(s), str(w)) + paths[i]
                npz = np.load(pathname)
                matrix[i, ids, idw] = np.average(
                    np.std(npz['avg_real_aoi'][npz['actual_slot'] == 1394][:, 100:], axis=1))
                if i > 0:
                    compare_reduce_rate.append(matrix[0, ids, idw] / matrix[i, ids, idw])
    minv = min(compare_reduce_rate)
    maxv = max(compare_reduce_rate)
    averagev = np.average(compare_reduce_rate)

    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    diff = [-0.3, -0.1, 0.1, 0.3]
    width = 0.2
    for idx, ax in enumerate(axs):
        for idx1 in range(4):
            ax.bar([i + diff[idx1] for i in range(len(workers))], matrix[idx1][idx, :], width=width,
                   color=plt.get_cmap('tab20')(idx1), label=labels[idx1], zorder=5)

        # ax.set_xlim(-1, len(workers))
        ax.set_xticks([i for i in range(len(workers))])
        ax.set_xticklabels(workers)
        ax.set_ylim(0, 12)
        ax.yaxis.grid(linestyle='--', zorder=0)
        ax.set_xlabel('worker number $M$')
        ax.set_ylabel('Standard deviation of global AoI')
        ax.set_title(titles[idx], y=-0.26)
        ax.legend()
        # ax.grid(linestyle='--')

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + "new_std_sen_work.tif")
    plt.savefig(save_path + "new_std_sen_work.jpg")


def plot_new_aoi_curve():
    save_name = 'new_aoi_curve'
    workers = [100, 500, 1000, 5000]
    # sensors = [500, 1000, 5000, 10000]
    labels = ['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP']
    titles = ['(a) Worker number $M$ = 100', '(b) Worker number $M$ = 500', '(c) Worker number $M$ = 1000',
              '(d) Worker number $M$ = 5000']
    upper_limits = [60, 54, 48, 18]
    step = [10, 9, 8, 3]
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(workers), 1394), dtype=np.float64)
    for i in range(4):
        for idw, w in enumerate(workers):
            pathname = 'save/t-drive_sen_5000_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999' \
                           .format(str(w)) + paths[i]
            npz = np.load(pathname)
            matrix[i, idw] = np.mean(npz['avg_real_aoi'][npz['actual_slot'] == 1394], axis=0)

    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    for idx, ax in enumerate(axs):
        for idx1 in range(4):
            ax.plot([i for i in range(1, 1395)], matrix[idx1, idx], color=plt.get_cmap('tab10')(idx1),
                    label=labels[idx1])

        ax.set_yticks(np.arange(0, upper_limits[idx] + 1, step[idx]))
        ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
        ax.set_xticklabels(['Feb. 3', 'Feb. 4', 'Feb. 5', 'Feb. 6', 'Feb. 7', 'Feb. 8'])
        ax.set_xlim(0, 1395)
        ax.set_ylim(0, upper_limits[idx])
        ax.set_xlabel('Date')
        ax.set_ylabel('Average global AoI (${\\Delta t}$)')
        ax.set_title(titles[idx], y=-0.26)
        ax.legend()
        ax.grid(linestyle='--')

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + save_name + ".tif")
    plt.savefig(save_path + save_name + ".jpg")


def reduce_rate():
    sum_average = []
    good_average = []
    bad_average = []
    for reduce in [0.5, 0.75, 0.875, 0.9375, 0.96875]:
        data = np.load(
            'save/t-drive_sen_5000_wkr_5000_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_reduce_{}.npz'.format(
                str(reduce)))
        [x, y] = [data['good_task_number'], data['bad_task_number']]
        sum_average.append(np.average(x + y))
        good_average.append(np.average(x))
        bad_average.append(np.average(y))

    a = 1


def new_plot_heat_map_aoi():
    # labels = ['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = []
    t = [i for i in range(1, 11)]
    for i in range(4):
        pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999' + paths[i]
        npz = np.load(pathname)
        matrix.append(pd.DataFrame(np.mean(npz['real_aoi_by_slot'], axis=0), index=t, columns=t))
    titles = ['(a) DRL-GAM', '(b) Greedy', '(c) Robin Round', '(d) CCPP']
    fig = plt.figure(figsize=(18, 4), dpi=300)
    axs = []
    for idx, i in enumerate(range(141, 145)):
        ax = fig.add_subplot(i)
        axs.append(ax)
        sns.heatmap(data=matrix[idx], ax=ax, annot=True, cmap="GnBu", vmax=1.0, vmin=0, fmt='.2f',
                    annot_kws={'fontsize': 4}, cbar_kws={'label': 'Average cell AoI (${\\Delta t}$)'})
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(titles[idx], y=-0.26)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + 'new_heat_aoi.tif')
    plt.savefig(save_path + 'new_heat_aoi.jpg')


def new_plot_heat_map_visit():
    # labels = ['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = []
    t = [i for i in range(1, 11)]
    maxvs = [3.0, 5.4, 1.8, 1.8]
    ticks = [[0, .5, 1, 1.5, 2, 2.5, 3], [0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
             [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]]
    for i in range(4):
        pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999' + paths[i]
        npz = np.load(pathname)
        temp = np.mean(npz['visit_time'], axis=0)
        matrix.append(pd.DataFrame(temp / np.sum(temp) * 100, index=t, columns=t))
    titles = ['(a) DRL-GAM', '(b) Greedy', '(c) Robin Round', '(d) CCPP']
    fig = plt.figure(figsize=(18, 4), dpi=300)
    axs = []
    for idx, i in enumerate(range(141, 145)):
        ax = fig.add_subplot(i)
        axs.append(ax)
        sns.heatmap(data=matrix[idx], ax=ax, annot=True, cmap="GnBu", vmax=maxvs[idx], vmin=0, fmt='.2f',
                    annot_kws={'fontsize': 4},
                    cbar_kws={'label': 'Access probability of the UAV (%)', 'ticks': ticks[idx]})
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(titles[idx], y=-0.26)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_path + 'new_visit_time.tif')
    plt.savefig(save_path + 'new_visit_time.jpg')


def new_pho_trust():
    save_name = save_path + 'new_pho'
    pho = [0.3, 0.5, 0.7, 0.9]
    mali = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    pho_datas = np.zeros(shape=(len(pho), len(mali)))
    good_datas = np.zeros(shape=(len(pho), 1394))
    bad_datas = np.zeros(shape=(len(pho), 1394))
    for i, p in enumerate(pho):
        for j, m in enumerate(mali):
            pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_pho_{}_mali_{}_new.npz' \
                .format(str(p), str(m))
            npz = np.load(pathname)
            pho_datas[i, j] = np.mean(npz['norm'][npz['actual_slot'] == 1394][:, 200:])
            # pho_datas.append(pd.DataFrame(temp / np.sum(temp) * 100, index=t, columns=t))
            if m == 0.5:
                good_datas[i] = np.mean(npz['good_trust'][npz['actual_slot'] == 1394], axis=0)
                bad_datas[i] = np.mean(npz['bad_trust'][npz['actual_slot'] == 1394], axis=0)

    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\\rho=0.3$', '$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']

    for idx, i in enumerate(pho):
        axl.plot(mali, pho_datas[idx], color=plt.get_cmap('tab10')(idx), label=names[idx])
        print(idx)
    axl.set_xlabel('Ratio of malicious workers')
    axl.set_ylabel('2-norm error of estimated and actual AoI')

    # axl.set_xlim(0, 1395)
    axl.set_ylim(0, 0.6)
    axl.legend()
    axl.set_title('(a) Effect of $\\rho$ on AoI estimation', y=-0.26)
    axl.grid(linestyle='--')

    axg = fig.add_subplot(122)

    names = ['$\\rho=0.3$', '$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']
    for idx, i in enumerate(names):
        axg.plot([i for i in range(1, 1395)], good_datas[idx], color=plt.get_cmap('Set1')(idx),
                 label='{}: normal'.format(i))
        axg.plot([i for i in range(1, 1395)], bad_datas[idx], color=plt.get_cmap('Set1')(idx), linestyle='--',
                 label='{}: malicious '.format(i))
        print(idx)
    axg.set_xlabel('Date')
    axg.set_ylabel('Trust value')
    axg.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    axg.set_xticklabels(['Feb. 3', 'Feb. 4', 'Feb. 5', 'Feb. 6', 'Feb. 7', 'Feb. 8'])
    axg.set_xlim(0, 1395)
    axg.set_ylim(0, 1)
    axg.legend()
    axg.set_title('(b)  Effect of $\\rho$ on trust of workers', y=-0.26)
    axg.grid(linestyle='--')
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_name + '.tif')
    plt.savefig(save_name + '.jpg')


def new_reduce_test():
    save_name = save_path + 'new_reduce'
    reduce = [0.5, 0.75, 0.875, 0.9375, 0.96875]
    mali = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    reduce_datas = np.zeros(shape=(len(reduce), len(mali)))
    task_datas = np.zeros(shape=(len(reduce), len(mali)))
    for i, p in enumerate(reduce):
        for j, m in enumerate(mali):
            pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_reduce_{}_mali_{}_pho_0.5_new.npz' \
                .format(str(p), str(m))
            npz = np.load(pathname)
            reduce_datas[i, j] = np.mean(npz['norm'][npz['actual_slot'] == 1394])
            task_datas[i, j] = np.mean((npz['good_task_number'] + npz['bad_task_number'])[npz['actual_slot'] == 1394])
            # pho_datas.append(pd.DataFrame(temp / np.sum(temp) * 100, index=t, columns=t))

    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\\Lambda=0.5$', '$\\Lambda=0.75$', '$\\Lambda=0.875$', '$\\Lambda=0.9375$', '$\\Lambda=0.96875$']

    for idx, i in enumerate(reduce):
        axl.plot(mali, reduce_datas[idx], color=plt.get_cmap('tab10')(idx), label=names[idx])
        print(idx)
    axl.set_xlabel('Ratio of malicious workers')
    axl.set_ylabel('2-norm error of estimated and actual AoI')
    # axl.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    # axl.set_xlim(0, 1395)
    # axl.set_ylim(0, 0.7)
    axl.legend()
    axl.set_title('(a) Effect of $\\Lambda$ on AoI estimation', y=-0.26)
    axl.grid(linestyle='--')

    axg = fig.add_subplot(122)

    for idx, i in enumerate(reduce):
        axg.plot(mali, task_datas[idx] / 1000, color=plt.get_cmap('tab10')(idx), label=names[idx])
        print(idx)
    axg.set_xlabel('Ratio of malicious workers')
    axg.set_ylabel('The number of assigned tasks ($\\times10^3$)')
    axg.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    # axg.set_xticklabels(['Feb. 3', 'Feb. 4', 'Feb. 5', 'Feb. 6', 'Feb. 7', 'Feb. 8'])
    # axg.set_xlim(0, 1395)
    # axg.set_ylim(0, 1)
    axg.legend()
    axg.set_title('(b) Effect of $\\Lambda$ on task assignment', y=-0.26)
    axg.grid(linestyle='--')
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    plt.savefig(save_name + '.tif')
    plt.savefig(save_name + '.jpg')


def new_mali_plot():
    save_name = 'new_mali_est_assign'
    # [random_assign, random_norm, test_assign, test_norm] = mali_compare()
    # reduce = [0.5, 0.75, 0.875, 0.9375, 0.96875]
    random = ['_random', '_new']
    mali = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    norms = np.zeros(shape=(len(random), len(mali)))
    good_task_datas = np.zeros(shape=(len(random), len(mali)))
    bad_task_datas = np.zeros(shape=(len(random), len(mali)))
    total_aoi = np.zeros(shape=(len(random), len(mali)))
    total_tasks = np.zeros(shape=(len(random), len(mali)))
    for i, p in enumerate(random):
        for j, m in enumerate(mali):
            pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_reduce_0.875_mali_{}_pho_0.5{}.npz' \
                .format(str(m), p)
            npz = np.load(pathname)
            norms[i, j] = np.mean(npz['norm'][npz['actual_slot'] == 1394])
            # total_aoi[i, j] = np.mean(npz['avg_real_aoi'][npz['actual_slot'] == 1394])
            total_aoi[i, j] = np.nanmean(npz['avg_real_aoi'][npz['actual_slot'] == 1394])
            good_task_datas[i, j] = np.nanmean(npz['good_task_number'][npz['actual_slot'] == 1394] / 1000)
            bad_task_datas[i, j] = np.nanmean(npz['bad_task_number'][npz['actual_slot'] == 1394] / 1000)
            total_tasks[i, j] = good_task_datas[i, j] + bad_task_datas[i, j]
            # pho_datas.append(pd.DataFrame(temp / np.sum(temp) * 100, index=t, columns=t))

    norm_reduce = np.average((norms[0] - norms[1]) / norms[0])
    total_aoi_reduce = np.average((total_aoi[0] - total_aoi[1]) / total_aoi[0])
    total_assign = np.average((total_tasks[0] - total_tasks[1]) / total_tasks[0])
    bad_task_reduce = np.average((bad_task_datas[0] - bad_task_datas[1]) / bad_task_datas[0])
    matrix = np.array([norms, total_aoi, good_task_datas, bad_task_datas])

    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    diff = [-0.02, 0.02]
    labels = ['Random', 'GMTA']
    titles = ['(a) Error performance', '(b) AoI performance', '(c) Tasks assigned to normal workers',
              '(d) Tasks assigned to malicious workers']
    x_label = 'Ratio of malicious workers'
    y_labels = ['2-Norm error of estimated and actual AoI', 'Average global AoI',
                'The number of assigned tasks ($\\times10^3$)', 'The number of assigned tasks ($\\times10^3$)']
    for idx, ax in enumerate(axs):
        for idx1 in range(2):
            ax.bar(np.array(mali) + diff[idx1], matrix[idx][idx1], width=0.04, color=plt.get_cmap('tab10')(idx1),
                   label=labels[idx1], zorder=5)

        # ax.set_yticks(np.arange(0, upper_limits[idx] + 1, step[idx]))
        # ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
        # ax.set_xticklabels(['Feb. 3', 'Feb. 4', 'Feb. 5', 'Feb. 6', 'Feb. 7', 'Feb. 8'])
        # ax.set_xlim(0, 1395)
        # ax.set_ylim(0, upper_limits[idx])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[idx])
        ax.set_title(titles[idx], y=-0.26)
        ax.legend()
        ax.yaxis.grid(linestyle='--', zorder=0)

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    # plt.savefig(save_path + save_name + ".tif")
    plt.savefig(save_path + save_name + ".jpg")


def plot_simulation():
    cleaner = DataCleaner()

    c = Cell.uniform_generator_with_position(cleaner, 5000)

    name = 'x' + str(cleaner.x_limit) + '_y' + str(cleaner.y_limit) + ('_uniform' if not cleaner.Norm else '_norm') \
           + ('_no_dataset' if cleaner.No_data_set_need else '_with_dataset')

    [sensor_x, sensor_y] = Sensor.get_all_locations()
    map_fig = plt.figure(figsize=(10, 8), dpi=250)
    ax = map_fig.add_subplot(111)
    np.random.seed(10)
    # if not cleaner.No_data_set_need:
    sp = cleaner.worker_coordinate().shape[0]
    sample_nodes = cleaner.worker_coordinate()[np.random.choice(sp, int(sp / 5), False), :]
    # work_p = ax.scatter(cleaner.worker_coordinate()[:, 0], cleaner.worker_coordinate()[:, 1], color='gray',
    #                     marker='o', s=0.01, alpha=0.4)
    work_p = ax.scatter(sample_nodes[:, 0], sample_nodes[:, 1], color='gray',
                        marker='.', s=0.01, alpha=0.4)

    work_p1 = ax.scatter([0], [0], color='gray',
                         marker='.', s=1, alpha=1)
    line_edge = None
    for row in c:
        for cell in row:
            line_edge, = cell.plot_cell(ax)

    sen_p = ax.scatter(sensor_x, sensor_y, color='royalblue', marker='o', s=2.5, alpha=0.75)
    uav_p = ax.scatter(cleaner.cell_coordinate[4, 2][0], cleaner.cell_coordinate[4, 2][1], color='r', marker='o', s=40)
    plt.xlim(cleaner.x_range[0] - cleaner.side_length, cleaner.x_range[1] + cleaner.side_length)
    plt.ylim(cleaner.y_range[0] - cleaner.side_length * 2 / sqrt(3),
             cleaner.y_range[1] + cleaner.side_length * 2 / sqrt(3))

    # rectangle
    # deploy_range, = ax.plot([cleaner.x_range[0], cleaner.x_range[1], cleaner.x_range[1], cleaner.x_range[0], cleaner.x_range[0]],
    #         [cleaner.y_range[0], cleaner.y_range[0], cleaner.y_range[1], cleaner.y_range[1], cleaner.y_range[0]],
    #         color='gray', linewidth=1.5, linestyle='--')

    if cleaner.Range_is_angle:
        plt.xlabel("经度(°)")
        plt.ylabel("纬度(°)")
    else:
        plt.xlabel("x(meter)")
        plt.ylabel("y(meter)")
    plt.legend((sen_p, uav_p, line_edge, work_p1), ("传感器节点", "无人机节点", "小区范围", "众包节点轨迹点"),
               loc='lower right', framealpha=1)
    if not os.path.exists('./figure/'):
        os.mkdir('./figure/')
    plt.savefig("./figure/{}.tif".format(name), bbox_inches='tight', )
    plt.savefig("./figure/{}.jpg".format(name), bbox_inches='tight', )
    print('ok')


def plot_slot_curve_worker_number_effect():
    rcParams['font.size'] = 14  # 设置字体大小
    map_fig = plt.figure(figsize=(12, 10), dpi=250)
    ax = map_fig.add_subplot(211)
    ax2 = map_fig.add_subplot(223)
    ax3 = map_fig.add_subplot(224)
    save_name = 'cell_10_in_one_slot_worker_number'
    file = '/home/huangshaobo/workspace/UPMA_random_worker_1/save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_{}_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/test/Test.npz'
    number = [250, 500, 1000, 2500, 5000, 10000]
    labels = ['众包节点数量 $M = {}$'.format(number[i]) for i in range(6)]
    files = [file.format(str(n)) for n in number]


    data = [np.average(np.load(name, allow_pickle=True)['avg_real_aoi'], axis=0) for name in files]
    avg_data = []
    peak_data = []
    for i in range(6):
        avg_data.append(np.average(np.load(files[i], allow_pickle=True)['avg_real_aoi']))
        temp_peak = []
        for di in range(1, 1393):
            if data[i][di] > data[i][di - 1] and data[i][di] > data[i][di + 1]:
                temp_peak.append(data[i][di])
        temp_peak.sort(reverse=True)
        peak_data.append(np.average(temp_peak))

    x = [i for i in range(1, 1395)]
    for idx in range(6):
        ax.plot(x, data[idx], color=plt.get_cmap('tab10')(idx), label=labels[idx])

    ax.set_xlim(0, 1395)
    # ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
    ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    ax.set_xticklabels(['2月3日', '2月4日', '2月5日', '2月6日', '2月7日', '2月8日'])
    ax.set_xlabel('日期')
    ax.set_ylabel('全局信息年龄($\Delta t$)')
    ax.legend(ncol=3, prop={'size': 10})
    ax.grid(linestyle='--', )
    ax.set_title('(a) 不同众包节点数量下的全局信息年龄', y=-0.3)

    ax2.bar([i for i in range(6)], avg_data, zorder=10)
    for i in range(6):
        ax2.text(i, avg_data[i] + 1, '%.3f'%avg_data[i], ha='center', zorder=11)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel('众包节点数量')
    ax2.set_ylabel('时间平均信息年龄($\Delta t$)')
    ax2.set_xticks([i for i in range(6)])
    ax2.set_xticklabels(number)
    ax2.set_title('(b) 众包节点数量对时间平均信息年龄的作用', y=-0.3)
    ax2.grid(linestyle='--', axis='y', zorder=5)


    ax3.bar([i for i in range(6)], peak_data, zorder=10)
    for i in range(6):
        ax3.text(i, peak_data[i] + 1, '%.3f'%peak_data[i], ha='center', zorder=11)
    ax3.set_ylim(0, 30)
    ax3.set_xticks([i for i in range(6)])
    ax3.set_xticklabels(number)
    ax3.set_xlabel('众包节点数量')
    ax3.set_ylabel('峰值平均信息年龄($\Delta t$)')
    ax3.set_title('(c) 众包节点数量对峰值平均信息年龄的作用', y=-0.3)
    ax3.grid(linestyle='--', axis='y', zorder=5)

    _250_500_avg_reduce = (avg_data[0] - avg_data[1]) / avg_data[0]
    _5000_10000_avg_reduce = (avg_data[4] - avg_data[5]) / avg_data[4]

    _250_500_peak_reduce = (peak_data[0] - peak_data[1]) / peak_data[0]
    _5000_10000_peak_reduce = (peak_data[4] - peak_data[5]) / peak_data[4]
    print("250 - 500 avg {} peak {}".format(_250_500_avg_reduce, _250_500_peak_reduce))
    print("5000 - 10000 avg {} peak {}".format(_5000_10000_avg_reduce, _5000_10000_peak_reduce))

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.35)
    plt.savefig("./figure/{}.tif".format(save_name), bbox_inches='tight', )
    plt.savefig("./figure/{}.jpg".format(save_name), bbox_inches='tight', )

def plot_slot_curve_cost_effect():
    rcParams['font.size'] = 14  # 设置字体大小
    map_fig = plt.figure(figsize=(12, 10), dpi=250)
    ax = map_fig.add_subplot(211)
    ax2 = map_fig.add_subplot(223)
    ax3 = map_fig.add_subplot(224)
    save_name = 'cell_10_in_one_slot_cost'
    file = '/home/huangshaobo/workspace/UPMA_random_worker_1/save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_{}_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/test/Test.npz'
    number = [250, 500, 1000, 2000, 4000, 8000]
    labels = ['总预算 $J_{\max} = 250$', '总预算 $J_{\max} = 500$', '总预算 $J_{\max} = 1000$', '总预算 $J_{\max} = 2000$', '总预算 $J_{\max} = 4000$', '总预算 $J_{\max} = 8000$']
    files = [file.format(str(n)) for n in number]

    data = [np.average(np.load(name, allow_pickle=True)['avg_real_aoi'], axis=0) for name in files]
    avg_data = []
    peak_data = []
    for i in range(6):
        avg_data.append(np.average(np.load(files[i], allow_pickle=True)['avg_real_aoi']))
        temp_peak = []
        for di in range(1, 1393):
            if data[i][di] > data[i][di - 1] and data[i][di] > data[i][di + 1]:
                temp_peak.append(data[i][di])
        temp_peak.sort(reverse=True)
        peak_data.append(np.average(temp_peak))
    x = [i for i in range(1, 1395)]
    for idx in range(6):
        ax.plot(x, data[idx], color=plt.get_cmap('tab10')(idx), label=labels[idx])

    ax.set_xlim(0, 1395)
    ax.set_ylim(0, 20)
    # ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
    ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    ax.set_xticklabels(['2月3日', '2月4日', '2月5日', '2月6日', '2月7日', '2月8日'])
    ax.set_xlabel('日期')
    ax.set_ylabel('全局信息年龄($\Delta t$)')
    ax.legend(ncol=3, prop={'size': 10})
    ax.grid(linestyle='--', )
    ax.set_title('(a) 不同总预算下的全局信息年龄', y=-0.3)

    ax2.bar([i for i in range(6)], avg_data, zorder=10)
    for i in range(6):
        ax2.text(i, avg_data[i] + 1, '%.3f'%avg_data[i], ha='center', zorder=11)
    ax2.set_ylim(0, 20)
    ax2.set_xlabel('总预算')
    ax2.set_ylabel('时间平均信息年龄($\Delta t$)')
    ax2.set_xticks([i for i in range(6)])
    ax2.set_xticklabels(number)
    ax2.set_title('(b) 总预算对时间平均信息年龄的作用', y=-0.3)
    ax2.grid(linestyle='--', axis='y', zorder=5)

    ax3.bar([i for i in range(6)], peak_data, zorder=10)
    for i in range(6):
        ax3.text(i, peak_data[i] + 1, '%.3f'%peak_data[i], ha='center', zorder=11)
    ax3.set_ylim(0, 20)
    ax3.set_xticks([i for i in range(6)])
    ax3.set_xticklabels(number)
    ax3.set_xlabel('总预算')
    ax3.set_ylabel('峰值平均信息年龄($\Delta t$)')
    ax3.set_title('(c) 总预算对峰值平均信息年龄的作用', y=-0.3)
    ax3.grid(linestyle='--', axis='y', zorder=5)

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.35)
    plt.savefig("./figure/{}.tif".format(save_name), bbox_inches='tight', )
    plt.savefig("./figure/{}.jpg".format(save_name), bbox_inches='tight', )

def plot_slot_curve_sen_effect():
    rcParams['font.size'] = 14  # 设置字体大小
    map_fig = plt.figure(figsize=(12, 10), dpi=250)
    ax = map_fig.add_subplot(211)
    ax2 = map_fig.add_subplot(223)
    ax3 = map_fig.add_subplot(224)
    save_name = 'cell_10_in_one_slot_sen'
    file = '/home/huangshaobo/workspace/UPMA_random_worker_1/save/x_10_y_10/t-drive/uniform_sensor/sen_{}_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/test/Test_greedy.npz'
    number = [1000, 2000, 3000, 4000, 5000, 6000]
    labels = ['传感器数量 $N = 1000$', '传感器数量 $N = 2000$', '传感器数量 $N = 3000$', '传感器数量 $N = 4000$', '传感器数量 $N = 5000$', '传感器数量 $N = 6000$']
    files = [file.format(str(n)) for n in number]

    data = [np.average(np.load(name, allow_pickle=True)['avg_real_aoi'], axis=0) for name in files]
    avg_data = []
    peak_data = []
    for i in range(6):
        avg_data.append(np.average(np.load(files[i], allow_pickle=True)['avg_real_aoi']))
        temp_peak = []
        for di in range(1, 1393):
            if data[i][di] > data[i][di - 1] and data[i][di] > data[i][di + 1]:
                temp_peak.append(data[i][di])
        temp_peak.sort(reverse=True)
        peak_data.append(np.average(temp_peak))
    x = [i for i in range(1, 1395)]
    for idx in range(6):
        ax.plot(x, data[idx], color=plt.get_cmap('tab10')(idx), label=labels[idx])

    ax.set_xlim(0, 1395)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 3, 6, 9, 12, 15, 18])
    # ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
    ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    ax.set_xticklabels(['2月3日', '2月4日', '2月5日', '2月6日', '2月7日', '2月8日'])
    ax.set_xlabel('日期')
    ax.set_ylabel('全局信息年龄($\Delta t$)')
    ax.legend(ncol=3, prop={'size': 10})
    ax.grid(linestyle='--', )
    ax.set_title('(a) 不同传感器数量下的全局信息年龄', y=-0.3)

    ax2.bar([i for i in range(6)], avg_data, zorder=10)
    for i in range(6):
        ax2.text(i, avg_data[i] + 1, '%.3f'%avg_data[i], ha='center', zorder=11)
    ax2.set_ylim(0, 15)
    ax2.set_yticks([0, 3, 6, 9, 12, 15])
    ax2.set_xlabel('传感器数量')
    ax2.set_ylabel('时间平均信息年龄($\Delta t$)')
    ax2.set_xticks([i for i in range(6)])
    ax2.set_xticklabels(number)
    ax2.set_title('(b) 传感器数量对时间平均信息年龄的作用', y=-0.3)
    ax2.grid(linestyle='--', axis='y', zorder=5)

    ax3.bar([i for i in range(6)], peak_data, zorder=10)
    for i in range(6):
        ax3.text(i, peak_data[i] + 1, '%.3f'%peak_data[i], ha='center', zorder=11)
    ax3.set_ylim(0, 15)
    ax3.set_xticks([i for i in range(6)])
    ax3.set_yticks([0, 3, 6, 9, 12, 15])
    ax3.set_xticklabels(number)
    ax3.set_xlabel('传感器数量')
    ax3.set_ylabel('峰值平均信息年龄($\Delta t$)')
    ax3.set_title('(c) 传感器数量对峰值平均信息年龄的作用', y=-0.3)
    ax3.grid(linestyle='--', axis='y', zorder=5)

    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.35)
    plt.savefig("./figure/{}.tif".format(save_name), bbox_inches='tight', )
    plt.savefig("./figure/{}.jpg".format(save_name), bbox_inches='tight', )

def plot_move_actions_4_1():
    rcParams['font.size'] = 12  # 设置字体大小
    class HandlerArrow(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            p = mpatches.FancyArrow(0, 0.5 * height, width, 0, width=3, color=np.array([253 / 255, 30 / 255, 30 / 255]),length_includes_head=True, shape='right')
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]
    cleaner = DataCleaner()

    c = Cell.uniform_generator_with_position(cleaner, 5000)

    name = 'cell_10_move_action_4_1'
    path_ = '/home/huangshaobo/workspace/UPMA_random_worker_1/save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/'
    files = ['test/Test.npz', 'compare/CCPP.npz', 'compare/Greedy.npz', 'compare/RR.npz']
    labels = ['(a) D3QN', '(b) CCPP', '(c) Greedy', '(d) RR']
    file_names = [path_ + f for f in files]
    mean_move_actions = []
    for i in range(4):
        sum_actions = 0
        move_actions_mtx = np.load(path_+files[i], allow_pickle=True)['move_action']
        temp_actions = {}
        for j in range(100):
            move_actions = move_actions_mtx[j]
            for key, val in move_actions.items():
                sum_actions += val
                temp_actions[key] = temp_actions.get(key, 0) + val

        for key, val in temp_actions.items():
            temp_actions[key] /= sum_actions
        mean_move_actions.append(temp_actions)

    move_actions = [np.load(name, allow_pickle=True)['move_action'][5] for name in file_names]

    [sensor_x, sensor_y] = Sensor.get_all_locations()
    map_fig = plt.figure(figsize=(12, 10.5), dpi=250)

    max_val_arrow = None
    sp = cleaner.worker_coordinate().shape[0]
    sample_nodes = cleaner.worker_coordinate()[np.random.choice(sp, int(sp / 5), False), :]
    global_max = 0
    for i in range(4):
        for (fx, fy, tx, ty), val in move_actions[i].items():
            global_max = max(val, global_max)
    norm = mpl.colors.LogNorm(vmin=1, vmax=100)
    cmap = mpl.cm.get_cmap('autumn_r')
    for idx, n in enumerate([221, 222, 223, 224]):
        ax = map_fig.add_subplot(n)


        charge_min = 9999
        charge_max = 0
        move_min = 9999
        move_max = 0
        # sum_move = 0
        charge_pos = 0
        for (fx, fy, tx, ty), val in move_actions[idx].items():
            if fx == tx and fy == ty:
                charge_min = min(charge_min, val)
                charge_max = max(charge_max, val)
                charge_pos += 1
            else:
                move_min = min(move_min, val)
                move_max = max(move_max, val)
                # sum_move += val

        color_min = np.array([1.0, 0.0, 0.0, 1.0])
        color_max = (np.array([1, 1, 1, 1]) - color_min) * 0.99 + color_min

        alpha_min = 0
        alpha_max = 1
        quiver_length = 1 / 2

        dot_min_scale = 5
        dot_max_scale = 120

        bias_num = 1 / 4  # 往返箭头的间距
        acbias_num = bias_num / 2
        bias_cof = {
            (0, 1, 0): np.array([0, acbias_num]),
            (1, 1, 0): np.array([0, acbias_num]),
            (0, -1, 0): np.array([0, -acbias_num]),
            (1, -1, 0): np.array([0, -acbias_num]),

            (0, 0, 1): np.array([-acbias_num / 2 * sqrt(3), acbias_num / 2]),
            (1, 1, 1): np.array([-acbias_num / 2 * sqrt(3), acbias_num / 2]),
            (0, -1, -1): np.array([acbias_num / 2 * sqrt(3), -acbias_num / 2]),
            (1, 0, -1): np.array([acbias_num / 2 * sqrt(3), -acbias_num / 2]),

            (0, -1, 1): np.array([-acbias_num / 2 * sqrt(3), -acbias_num / 2]),
            (1, 0, 1): np.array([-acbias_num / 2 * sqrt(3), -acbias_num / 2]),
            (0, 0, -1): np.array([acbias_num / 2 * sqrt(3), acbias_num / 2]),
            (1, 1, -1): np.array([acbias_num / 2 * sqrt(3), acbias_num / 2]),
        }

        for row in c:
            for cell in row:
                cell.plot_cell(ax, 1)

        # work_p = ax.scatter(cleaner.worker_coordinate()[:, 0], cleaner.worker_coordinate()[:, 1], color='gray',
        #                     marker='o', s=0.01, alpha=0.4)
        work_p = ax.scatter(sample_nodes[:, 0], sample_nodes[:, 1], color='gray',
                            marker='.', s=0.01, alpha=0.05)

        # sen_p = ax.scatter(sensor_x, sensor_y, color='royalblue', marker='o', s=1, alpha=0.5)
        # ax.scatter(0, 0, color='royalblue', marker='o', s=40, alpha=0.5, label='传感器节点')
        ax.scatter(0, 0, color='gray', marker='o', s=40, alpha=0.5, label='众包节点轨迹点')
        # uav_p = ax.scatter(cleaner.cell_coordinate[4, 2][0], cleaner.cell_coordinate[4, 2][1], color='r', marker='o', s=40)
        plt.xlim(cleaner.x_range[0] - cleaner.side_length, cleaner.x_range[1] + cleaner.side_length)
        plt.ylim(cleaner.y_range[0] - cleaner.side_length * 2 / sqrt(3),
                 cleaner.y_range[1] + cleaner.side_length * 2 / sqrt(3))

        charge_positions = np.empty(shape=(charge_pos, 2), dtype=np.float64)
        dot_sizes = np.empty(shape=(charge_pos,), dtype=np.float64)
        charge_cnt = 0
        for (fx, fy, tx, ty), val in move_actions[idx].items():
            if fx == tx and fy == ty:
                # charge_positions[charge_cnt] = cleaner.cell_coordinate[fx, fy]
                # dot_sizes[charge_cnt] = (val - charge_min + 1) / charge_max * (dot_max_scale - dot_min_scale) + dot_min_scale
                # charge_cnt+=1
                continue
            begin_cell_coor = cleaner.cell_coordinate[fx, fy]
            end_cell_coor = cleaner.cell_coordinate[tx, ty]

            vector = end_cell_coor - begin_cell_coor
            y_odd = fy % 2
            quiver_begin = begin_cell_coor + vector * (1 - quiver_length) / 2 + bias_cof.get(
                (y_odd, tx - fx, ty - fy)) * cleaner.side_length
            quiver_end = vector * quiver_length
            # print(val)
            # scale_size = (val - move_min) / move_max * (quiver_max_scale - quiver_min_scale) + quiver_min_scale

            # if idx == 3:
            #     move_min = 0
            # color = color_max - sqrt(1 - (1 - val / global_max) ** 2) * (color_max - color_min)
            # alpha = sqrt(1 - (1 - log2(val - move_min + 1) / log2(move_max)) ** 2) * (alpha_max - alpha_min) + alpha_min
            temp_arrow = ax.arrow(quiver_begin[0], quiver_begin[1], quiver_end[0], quiver_end[1], width=cleaner.side_length / 5, head_length=quiver_length * cleaner.side_length / 3 * 2, head_width=cleaner.side_length / 5 * 2,
                                  length_includes_head=True, shape='right', edgecolor='k', facecolor=cmap(norm(val)), linewidth=0.1, zorder=7)
            if val == global_max:
                max_val_arrow = temp_arrow
            #     temp_arrow = ax.arrow(quiver_begin[0], quiver_begin[1], quiver_end[0], quiver_end[1], width=20,
            #                           length_includes_head=True, shape='right', color=color, zorder=10, label='移动轨迹')
        # charge_p = ax.scatter(charge_positions[:,0], charge_positions[:,1], s=dot_sizes, marker='P', c='k', label='无人机充电小区')
            # break
        # quiver_p = ax.quiver(-1000, -1000, 1, 1, angles='xy', scale_units='xy', scale=1)
        # rectangle
        # deploy_range, = ax.plot(
        #     [cleaner.x_range[0], cleaner.x_range[1], cleaner.x_range[1], cleaner.x_range[0], cleaner.x_range[0]],
        #     [cleaner.y_range[0], cleaner.y_range[0], cleaner.y_range[1], cleaner.y_range[1], cleaner.y_range[0]],
        #     color='gray', linewidth=1.5, linestyle='--')
        ax.set_title(labels[idx], y=-0.23, fontsize=16)

        if cleaner.Range_is_angle:
            plt.xlabel("经度(°)")
            plt.ylabel("纬度(°)")
        else:
            plt.xlabel("x(meter)")
            plt.ylabel("y(meter)")

        # chinese_font


    # plt.legend((sen_p, uav_p, line_edge, deploy_range), ("Sensor nodes", "UAV", "Cell edges", "Deployment scope of sensors"), loc='lower right')
    h, l = plt.gca().get_legend_handles_labels()
    h.append(max_val_arrow)
    l.append('无人机移动轨迹')
    map_fig.legend(h, l, handler_map={mpatches.FancyArrow: HandlerArrow()}, loc='lower center', ncol=4, framealpha=1, fontsize=18)

    if not os.path.exists('./figure/'):
        os.mkdir('./figure/')
    map_fig.subplots_adjust(top=0.92, bottom=0.15, wspace=0.2, hspace=0.23, right=0.9)
    ax2 = plt.axes([0.92, 0.15, 0.03, 0.77])
    ax2.set_title('移动频次')

    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                orientation='vertical')
    # plt.tight_layout()
    plt.savefig("./figure/{}.tif".format(name), bbox_inches='tight', )
    plt.savefig("./figure/{}.jpg".format(name), bbox_inches='tight', )

    print('ok')

def plot_task_assignment_curve():
    fig = plt.figure(figsize=(12, 7.5), dpi=250)
    aoi_axs = [fig.add_subplot(i) for i in [231, 232, 233]]
    cpu_axs = [fig.add_subplot(i) for i in [234, 235, 236]]

    wkn = [250, 500, 750, 1000, 1250, 1500]
    sen = [1000, 2000, 3000, 4000, 5000, 6000]
    cst = [500, 1000, 1500, 2000, 2500, 3000]

    names = ['greedy', 'g-greedy', 'random', 'genetic']
    label = ['SSF-HO', 'g-Greedy', 'Random', 'Genetic']
    aoi_data = np.zeros(shape=(3, 4, 6), dtype=np.float64)
    cpu_data = np.zeros(shape=(3, 4, 6), dtype=np.float64)
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    for i in range(3):
        for j in range(4):
            for k in range(6):
                nop = ''
                if j == 1:
                    nop = '_nop'
                if i == 0:
                    path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_{}_cst_2000_epi_0_bat_256_lr_0_gama_0_epd_0.99999/compare/RR{}_{}.npz'.format(wkn[k], nop, names[j])
                elif i == 1:
                    path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_{}_wkr_1000_cst_2000_epi_0_bat_256_lr_0_gama_0_epd_0.99999/compare/RR{}_{}.npz'.format(sen[k], nop, names[j])
                else:
                    path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_{}_epi_0_bat_256_lr_0_gama_0_epd_0.99999/compare/RR{}_{}.npz'.format(cst[k], nop, names[j])

                aoi_data[i, j, k] = np.average(np.load(path, allow_pickle=True)['avg_real_aoi'])
                cpu_data[i, j, k] = np.average(np.load(path, allow_pickle=True)['task_assignment_time'])

    cmap = plt.get_cmap('tab10')
    x1_lbs = ['众包节点数量', '传感器节点数量', '总预算']
    titles1 = ['(a) 众包节点数量对信息年龄的作用', '(b) 传感器节点数量对信息年龄的作用', '(c) 总预算对信息年龄的作用']
    titles2 = ['(d) 众包节点数量对CPU时间的作用', '(e) 传感器节点数量对CPU时间的作用', '(f) 总预算对CPU时间的作用']
    for i, (ax1, ax2) in enumerate(zip(aoi_axs, cpu_axs)):
        if i == 0:
            x = wkn
        elif i == 1:
            x = sen
        else:
            x = cst
        for j, name in enumerate(names):
            ax1.plot(x, aoi_data[i][j], label=label[j], color=cmap(j))

            ax1.set_xlabel(x1_lbs[i], fontdict={'size': 12})
            ax1.set_xticks(x)
            ax1.grid(linestyle='--')
            ax1.set_title(titles1[i], y=-0.30,fontdict={'size': 14})

            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(11)
                # specify integer or one of preset strings, e.g.
                # tick.label.set_fontsize('x-small')
                # tick.label.set_rotation('vertical')


            ax2.plot(x, cpu_data[i][j], label=label[j], color=cmap(j))
            for tex_idx in range(6):
                val = cpu_data[i][j][tex_idx] * 1.2
                text_data = '%.4f'%cpu_data[i][j][tex_idx] if cpu_data[i][j][tex_idx] < 1 else '%.4g'%cpu_data[i][j][tex_idx]
                ax2.text(x[tex_idx], val, text_data, fontsize=6.5, ha='center')
            # ax2.text(idx[i], avg_data[idx[i]] + 1, '%.2f' % avg_data[idx[i]], fontsize=7, ha='center')
            ax2.set_xlabel(x1_lbs[i],fontdict={'size': 12})
            ax2.set_xticks(x)
            ax2.set_yscale('log')
            if i == 1:
                ax2.set_ylim(0.001, 100)
            else:
                ax2.set_ylim(0.0001, 100)
            ax2.set_title(titles2[i], y=-0.30, fontdict={'size': 14})
            ax2.grid(linestyle='--')
            for tick in ax2.xaxis.get_major_ticks():
                tick.label.set_fontsize(11)
            for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(11)
            # ax2.set_ylim(1e-4, 1000)

    cpu_ggreedy_data = cpu_data[:, 1, :]
    cpu_ssf_ho_data = cpu_data[:, 0, :]

    cpu_reduce = (cpu_ggreedy_data - cpu_ssf_ho_data) / cpu_ggreedy_data

    print("cpu reduce min {} max {} average {}".format(np.min(cpu_reduce), np.max(cpu_reduce), np.average(cpu_reduce)))

    aoi_ggreedy_data = aoi_data[:, 1, :]
    aoi_ssf_ho_data = aoi_data[:, 0, :]

    aoi_diff = np.abs(aoi_ggreedy_data - aoi_ssf_ho_data) / aoi_ggreedy_data
    print("aoi diff min {} max {} average {}".format(np.min(aoi_diff), np.max(aoi_diff), np.average(aoi_diff)))


    # plt.grid('--')
    aoi_axs[0].legend(fontsize=13)
    aoi_axs[0].set_ylabel('时间平均信息年龄($\Delta t$)', fontdict={'size': 12})
    cpu_axs[0].set_ylabel('CPU时间(s)', fontdict={'size': 12})
    plt.subplots_adjust(bottom=0.1, wspace=0.2, hspace=0.35)
    plt.savefig("./figure/cell_10_task_assignment.tif", bbox_inches='tight', )
    plt.savefig("./figure/cell_10_task_assignment.jpg", bbox_inches='tight', )


def plot_multi_status():
    fig = plt.figure(figsize=(12, 7), dpi=250)
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(223)
    # ax3 = fig.add_subplot(224)
    methods = ['Baseline: RR', 'Baseline: AoI-BBA + SSF-HO', 'Baseline: g-Greedy', 'Baseline: AoI-BBA + Random', 'D3QN + AoI-BBA + SSF-HO', 'D3QN + g-Greedy', 'D3QN + AoI-BBA + Random', 'RR + AoI-BBA + SSF-HO',
               'RR + g-Greedy', 'RR + AoI-BBA + Random', 'Greedy + AoI-BBA + SSF-HO', 'Greedy + g-Greedy', 'Greedy + AoI-BBA + Random']
    tsk_mtd = ['greedy', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random']    # 空表示随意，反正也不会用上
    wkr_num = [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    no_uav = [False, True, True, True, False, False, False, False, False, False, False, False, False]
    cmp = [True, False, False, False, False, False, False, True, True, True, True, True, True]
    cmp_mth = ['RR', '', '', '', '', '', '', 'RR', 'RR', 'RR', 'Greedy', 'Greedy', 'Greedy']
    over_state = [True, True, True, True, False, False, False, True, True, True, True, True, True]
    paths = []
    datas = []
    peak_data = []
    avg_data = []
    x = [i for i in range(1, 1395)]
    array_idx = [0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 4]
    idx = [5, 4, 6, 8, 7, 9, 11, 10, 12, 0, 2, 1, 3]
    axins = ax1.inset_axes((0.05, 0.55, 0.3, 0.3))
    axins1 = ax1.inset_axes((0.4, 0.7, 0.25, 0.25))
    for i in range(13):
        k = idx[i]
        path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_{}_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/'.format(str(wkr_num[k]))
        if cmp[k]:
            path += 'compare/' + cmp_mth[k]
        else:
            path += 'test/Test'
        path += ('_' + tsk_mtd[k])
        if no_uav[k]:
            path += '_no_uav'
        path += '.npz'
        temp_data = np.load(path, allow_pickle=True)['avg_real_aoi']
        datas.append(np.average(temp_data, axis=0))
        avg_data.append(np.average(temp_data))
        temp_peak = []
        for di in range(1, 1393):
            if datas[-1][di] > datas[-1][di - 1] and datas[-1][di] > datas[-1][di + 1]:
                temp_peak.append(datas[-1][di])
        peak_data.append(np.average(temp_peak))
        ax1.plot(x, datas[-1], label=methods[k], linewidth=1.5, color=plt.get_cmap('tab20')(i))
        axins.plot(x, datas[-1], linewidth=2, color=plt.get_cmap('tab20')(i))
        axins1.plot(x, datas[-1], linewidth=1.5, color=plt.get_cmap('tab20')(i))

    aoi_diff = np.abs(datas[0] - datas[1]) / datas[0]
    print("g-greedy and ssf-ho diff min {} max {} avg {}".format(np.min(aoi_diff), np.max(aoi_diff), np.average(aoi_diff)))

    begin = 1240
    end = 1395
    line_y = 53
    ax1.text(end - 5, line_y + 6, '平均51.92个工作周期', ha='right', va='center', fontsize=11)
    ax1.plot([1100, end], [line_y, line_y], linestyle='dashed', color='k')
    ax1.set_xlim(0, 1395)
    ax1.set_ylim(0, 175)
    ax1.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    ax1.set_xticklabels(['2月3日', '2月4日', '2月5日', '2月6日', '2月7日', '2月8日'])
    ax1.set_yticks([0, 25, 50, 75, 100, 125, 150, 175])

    axins.set_xticks([324.694189, 380.882784, 437.471379, 494.059974, 550.648569])
    axins.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'])
    axins.set_yticks([5, 10, 15, 20, 25, 30])
    for tick in axins.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    for tick in axins.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)

    axins.set_xlim(324.694189, 550.648569)
    axins.set_ylim(5, 30)



    axins1.set_xlim(380.882784, 437.471379)
    axins1.set_ylim(7.5, 10)

    dis_num = 4
    dis_ = (437.471379 - 380.882784) / dis_num
    axins1.set_xticks([380.882784 + dis_ * i for i in range(dis_num + 1)])
    if dis_num == 6:
        axins1.set_xticklabels(['6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00'])
    else:
        axins1.set_xticklabels(['6:00', '7:30', '9:00', '10:30', '12:00'])

    # axins1.yaxis.set_label_position("right")
    # axins1.yaxis.tick_right()

    axins1.set_yticks([7.5, 8, 8.5, 9, 9.5, 10])
    # axins1.set_xticklabels(['18:00', '19:00', '20:00', '21:00', '22:00'])

    for tick in axins1.xaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    for tick in axins1.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)
    ax1.grid(linestyle='--')
    # axins.grid(linestyle='--')
    # axins1.grid(linestyle='--')
    mark_inset(ax1, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1, zorder=10)

    mark_inset(axins, axins1, loc1=2, loc2=4, fc="none", ec='k', lw=1, zorder=10)

    # ax2.bar([i for i in range(13)], avg_data)
    # ax3.bar([i for i in range(13)], peak_data)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('全局信息年龄($\Delta t$)')
    plt.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.35)
    fig.legend(loc='lower center', ncol=3, framealpha=1, fontsize=13)
    plt.savefig("./figure/cell_10_multi_comp.tif", bbox_inches='tight', )
    plt.savefig("./figure/cell_10_multi_comp.jpg", bbox_inches='tight', )

def plot_multi_peak_avg_bar():
    rcParams['font.size'] = 10  # 设置字体大小
    fig = plt.figure(figsize=(9, 4), dpi=250)
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)
    methods = ['Baseline: RR', 'Baseline: AoI-BBA + SSF-HO', 'Baseline: g-Greedy', 'Baseline: AoI-BBA + Random',
               'D3QN + AoI-BBA + SSF-HO', 'D3QN + g-Greedy', 'D3QN + AoI-BBA + Random', 'RR + AoI-BBA + SSF-HO',
               'RR + g-Greedy', 'RR + AoI-BBA + Random', 'Greedy + AoI-BBA + SSF-HO', 'Greedy + g-Greedy',
               'Greedy + AoI-BBA + Random']
    tsk_mtd = ['greedy', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random',
               'greedy', 'g-greedy', 'random']  # 空表示随意，反正也不会用上
    wkr_num = [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    no_uav = [False, True, True, True, False, False, False, False, False, False, False, False, False]
    cmp = [True, False, False, False, False, False, False, True, True, True, True, True, True]
    cmp_mth = ['RR', '', '', '', '', '', '', 'RR', 'RR', 'RR', 'Greedy', 'Greedy', 'Greedy']
    over_state = [True, True, True, True, False, False, False, True, True, True, True, True, True]
    paths = []
    datas = []
    peak_data = []
    avg_data = []
    x = [i for i in range(1, 1395)]
    # array_idx = [0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 4]
    idx = [5, 4, 6, 8, 7, 9, 11, 10, 12, 0, 2, 1, 3]
    # axins = ax1.inset_axes((0.05, 0.55, 0.3, 0.3))
    # axins1 = ax1.inset_axes((0.4, 0.7, 0.25, 0.25))
    for i in range(13):
        k = idx[i]
        path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_{}_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/'.format(
            str(wkr_num[k]))
        if cmp[k]:
            path += 'compare/' + cmp_mth[k]
        else:
            path += 'test/Test'
        path += ('_' + tsk_mtd[k])
        if no_uav[k]:
            path += '_no_uav'
        path += '.npz'
        temp_data = np.load(path, allow_pickle=True)['avg_real_aoi']
        datas.append(np.average(temp_data, axis=0))
        avg_data.append(np.average(temp_data))
        temp_peak = []
        for di in range(1, 1393):
            if datas[-1][di] > datas[-1][di - 1] and datas[-1][di] > datas[-1][di + 1]:
                temp_peak.append(datas[-1][di])
        temp_peak.sort(reverse=True)
        peak_data.append(np.average(temp_peak))
        # ax1.plot(x, datas[-1], label=methods[k], linewidth=1.5, color=plt.get_cmap('tab20')(i))
        # axins.plot(x, datas[-1], linewidth=2, color=plt.get_cmap('tab20')(i))
        # axins1.plot(x, datas[-1], linewidth=1.5, color=plt.get_cmap('tab20')(i))

    ssf_ho_avg_diff = [(avg_data[i] - avg_data[1])/avg_data[i] for i in range(13)]
    ssf_ho_peak_diff = [(peak_data[i] - peak_data[1]) / peak_data[i] for i in range(13)]

    joint_diff = ssf_ho_avg_diff[2:9]
    non_joint_diff = ssf_ho_avg_diff[9:]

    joint_peak_diff = ssf_ho_peak_diff[2:9]
    non_joint_peak_diff = ssf_ho_peak_diff[9:]

    print("avg joint diff min {} max {} avg {}".format(np.min(joint_diff), np.max(joint_diff), np.average(joint_diff)))
    print("avg non joint diff min {} max {} avg {}".format(np.min(non_joint_diff), np.max(non_joint_diff), np.average(non_joint_diff)))

    print("peak joint diff min {} max {} avg {}".format(np.min(joint_peak_diff), np.max(joint_peak_diff), np.average(joint_peak_diff)))
    print("peak non joint diff min {} max {} avg {}".format(np.min(non_joint_peak_diff), np.max(non_joint_peak_diff), np.average(non_joint_peak_diff)))


    ax2.bar([i for i in range(13)], avg_data, zorder=10)
    ax3.bar([i for i in range(13)], peak_data, zorder=10)

    # fig.legend(loc='lower center', ncol=3, framealpha=1, fontsize=13)
    ticks = [methods[idx[i]] for i in range(13)]
    ax2.set_ylabel('时间平均信息年龄($\Delta t$)')
    ax2.set_xticks([i for i in range(13)])
    ax2.set_xticklabels(ticks, rotation=75, ha='right')
    ax2.set_xlabel('策略组合')
    ax2.set_title('(a) 不同策略组合下的时间平均信息年龄', y = -0.85)
    for i in range(13):
        ax2.text(idx[i], avg_data[idx[i]] + 1, '%.2f' %avg_data[idx[i]], fontsize=7, ha='center')

    ax2.grid(linestyle='--', axis='y', zorder=5)
    ax3.set_ylabel('峰值平均信息年龄($\Delta t$)')
    ax3.set_xticks([i for i in range(13)])
    ax3.set_title('(b) 不同策略组合下的峰值平均信息年龄', y = -0.85)
    ax3.set_xticklabels(ticks, rotation=75, ha='right')
    ax3.set_xlabel('策略组合')
    for i in range(13):
        ax3.text(idx[i], peak_data[idx[i]] + 1, '%.2f' %peak_data[idx[i]], fontsize=7, ha='center')
    ax3.grid(linestyle='--', axis='y', zorder=5)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)
    # plt.tight_layout()

    plt.savefig("./figure/cell_10_multi_comp_bar.tif", bbox_inches='tight', )
    plt.savefig("./figure/cell_10_multi_comp_bar.jpg", bbox_inches='tight', )

def plot_norm_and_trust_bar():
    rcParams['font.size'] = 10  # 设置字体大小
    fig = plt.figure(figsize=(9, 4), dpi=250)
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)

    norm_data = np.zeros(shape=(5, 9), dtype=np.float64)
    norm_data1 = np.zeros(shape=(5, 1394), dtype=np.float64)
    good_trust = np.zeros(shape=(7, 1394), dtype=np.float64)
    bad_trust = np.zeros(shape=(7, 1394), dtype=np.float64)
    for id1, pho in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        for id2, mali in enumerate([.1, .2, .3, .4, .5, .6, .7]):
            path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/compare/RR_pho_{}_mali_{}_greedy.npz'.format(pho, mali)
            path_slot = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/compare/RR_episode_pho_{}_mali_{}_greedy.csv'.format(pho, mali)

            data = np.load(path, allow_pickle=True)
            slot = pd.read_csv(path_slot)['slot number'].to_numpy()
            norm_data[id1, id2] = np.average(data['norm'][slot == 1394])

            if pho == 0.5:
                good_trust[id2] = np.average(data['good_task_number'][slot==1394], axis=0)
                bad_trust[id2] = np.average(data['bad_task_number'][slot==1394], axis=0)
            # if mali == 0.4:
            #     norm_data1[id1] = np.average(data, axis=0)

    a=1

def plot_assignment_curve():
    rcParams['font.size'] = 10  # 设置字体大小
    fig = plt.figure(figsize=(10, 4), dpi=250)
    ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(212)
    # ax2 = fig.add_subplot(121)
    # ax3 = fig.add_subplot(122)

    # norm_data = np.zeros(shape=(5, 9), dtype=np.float64)
    total_assignment = np.zeros(shape=(7, 1394), dtype=np.float64)
    good_assignment = np.zeros(shape=(7, 1394), dtype=np.float64)
    bad_assignment = np.zeros(shape=(7, 1394), dtype=np.float64)
    # good_trust = np.zeros(shape=(7, 1394), dtype=np.float64)
    # bad_trust = np.zeros(shape=(7, 1394), dtype=np.float64)
    names = ['恶意用户占比10%', '恶意用户占比20%','恶意用户占比30%','恶意用户占比40%','恶意用户占比50%','恶意用户占比60%','恶意用户占比70%',]
    x = []
    y = []
    z = []
    for id2, mali in enumerate([.1, .2, .3, .4, .5, .6, .7]):
        path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/test/Test_pho_0.5_mali_{}_greedy.npz'.format(mali)
        path_slot = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_1000_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/test/episode_pho_0.5_mali_{}_greedy.csv'.format(mali)

        data = np.load(path, allow_pickle=True)
        slot = pd.read_csv(path_slot)['slot number'].to_numpy()
        good_assignment[id2] = np.average(data['good_task_number'][slot==1394], axis=0)
        bad_assignment[id2] = np.average(data['bad_task_number'][slot==1394], axis=0)
        # good_trust[id2] = np.average(data['good_trust'][slot==1394], axis=0)
        # bad_trust[id2] = np.average(data['bad_trust'][slot==1394], axis=0)
        total_assignment[id2] = good_assignment[id2] + bad_assignment[id2]
        temp_x = np.arange(1, 1395, 1)
        x.append(temp_x[total_assignment[id2] > 0])
        y.append(bad_assignment[id2][x[-1]-1] / total_assignment[id2][x[-1]-1])
        # if mali == 0.4:
        #     norm_data1[id1] = np.average(data, axis=0)
        ax.plot(x[-1], y[-1],  color=plt.get_cmap('tab10')(id2), label=names[id2])
        # ax2.plot(good_trust[id2], color=plt.get_cmap('tab20')(id2 * 2))
        # ax2.plot(bad_trust[id2], color=plt.get_cmap('tab20')(id2 * 2 + 1))

        stable_val = np.average(y[-1][x[-1] > 776.602949])
        fist_val = y[-1][0]
        reduce = (fist_val - stable_val) / fist_val
        z.append(reduce)
        print(reduce)

    print('average {}'.format(np.average(z)))
    # ax2.set_xlim(0, 400)
    ax.set_xlim(0, 1395)
    ax.set_xticks([98.739809, 324.694189, 550.648569, 776.602949, 1002.557329, 1228.5117089999999])
    ax.set_xticklabels(['2月3日', '2月4日', '2月5日', '2月6日', '2月7日', '2月8日'])
    ax.set_xlim(0, 1395)
    ax.set_xlabel('日期')
    ax.set_ylabel('分配给恶意用户的任务占比')
    ax.grid(linestyle='--')
    ax.legend()



    plt.savefig("./figure/cell_10_task_assignment_number.tif", bbox_inches='tight', )
    plt.savefig("./figure/cell_10_task_assignment_number.jpg", bbox_inches='tight', )

def plot_multi_mali_assignment_bar():
    rcParams['font.size'] = 10  # 设置字体大小
    fig = plt.figure(figsize=(9, 4), dpi=250)
    ax2 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122)
    methods = ['Baseline: AoI-BBA + SSF-HO', 'Baseline: g-Greedy', 'Baseline: AoI-BBA + Random',
               'D3QN + AoI-BBA + SSF-HO', 'D3QN + g-Greedy', 'D3QN + AoI-BBA + Random', 'RR + AoI-BBA + SSF-HO',
               'RR + g-Greedy', 'RR + AoI-BBA + Random', 'Greedy + AoI-BBA + SSF-HO', 'Greedy + g-Greedy',
               'Greedy + AoI-BBA + Random']
    tsk_mtd = ['greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random', 'greedy', 'g-greedy', 'random',
               'greedy', 'g-greedy', 'random']  # 空表示随意，反正也不会用上
    wkr_num = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    no_uav = [True, True, True, False, False, False, False, False, False, False, False, False]
    cmp = [False, False, False, False, False, False, True, True, True, True, True, True]
    cmp_mth = ['', '', '', '', '', '', 'RR', 'RR', 'RR', 'Greedy', 'Greedy', 'Greedy']
    # over_state = [True, True, True, True, False, False, False, True, True, True, True, True, True]
    paths = []
    data = []
    peak_data = []
    avg_data = []
    normal_cost = []
    total_cost = []
    mali_cost = []
    x = [i for i in range(1, 1395)]
    # array_idx = [0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 4]
    idx = np.array([5, 4, 6, 8, 7, 9, 11, 10, 12, 2, 1, 3]) - 1
    # axins = ax1.inset_axes((0.05, 0.55, 0.3, 0.3))
    # axins1 = ax1.inset_axes((0.4, 0.7, 0.25, 0.25))

    for i in range(12):
        k = idx[i]
        path = root_path + 'save/x_10_y_10/t-drive/uniform_sensor/sen_5000_wkr_{}_cst_2000_epi_600_bat_256_lr_0.001_gama_0.9_epd_0.99999/'.format(
            str(wkr_num[k]))
        if cmp[k]:
            path += 'compare/' + cmp_mth[k]
        else:
            path += 'test/Test'
        path += ('_' + tsk_mtd[k])
        if no_uav[k]:
            path += '_no_uav'
        path += '.npz'
        temp_data = np.load(path, allow_pickle=True)
        good_data = temp_data['good_task_number']
        bad_data = temp_data['bad_task_number']
        total_data = good_data + bad_data
        avg = np.average(bad_data[total_data > 0] / total_data[total_data > 0])
        data.append(avg)


        good_cost = temp_data['good_assignment']
        bad_cost = temp_data['bad_assignment']
        total_data = good_cost + bad_cost

        normal_cost.append(np.average(good_cost[total_data > 0]))
        mali_cost.append(np.average(bad_cost[total_data > 0]))
        total_cost.append(normal_cost[-1] + mali_cost[-1])

        # ax1.plot(x, datas[-1], label=methods[k], linewidth=1.5, color=plt.get_cmap('tab20')(i))
        # axins.plot(x, datas[-1], linewidth=2, color=plt.get_cmap('tab20')(i))
        # axins1.plot(x, datas[-1], linewidth=1.5, color=plt.get_cmap('tab20')(i))

    reduce = np.zeros(shape=(12), dtype=np.float64)
    for i in range(12):
        comp_idx = int(int(i / 3) * 3) + 2
        reduce[idx[i]] = (data[idx[comp_idx]] - data[idx[i]]) / data[idx[comp_idx]]
    print(reduce)
    ax2.bar([i for i in range(12)], data, zorder=10)
    ax3.bar([i for i in range(12)], normal_cost, zorder=10, label='向正常用户分配的预算')
    ax3.bar([i for i in range(12)], mali_cost, bottom=normal_cost, zorder=11, label='向恶意用户分配的预算')
    # fig.legend(loc='lower center', ncol=3, framealpha=1, fontsize=13)
    ticks = [methods[idx[i]] for i in range(12)]
    ax2.set_ylabel('向恶意用户分配的任务占比')
    ax2.set_xticks([i for i in range(12)])
    ax2.set_xticklabels(ticks, rotation=75, ha='right')
    ax2.set_xlabel('策略组合')
    ax2.set_title('(a) 不同策略组合下向恶意用户分配的任务占比', y = -0.85)
    ax2.set_ylim(0, 0.2)
    for i in range(12):
        ax2.text(idx[i], data[idx[i]] + 0.005, '%.4f' %data[idx[i]], fontsize=7, ha='center', va='center')

    ax2.grid(linestyle='--', axis='y', zorder=5)


    ax3.set_ylabel('预算')
    ax3.set_xticks([i for i in range(12)])
    ax3.set_title('(b) 不同策略组合下向正常和恶意用户分配的预算', y = -0.85)
    ax3.set_xticklabels(ticks, rotation=75, ha='right')
    ax3.set_xlabel('策略组合')
    ax3.set_ylim(0, 2000)
    plt.legend().set_zorder(14)
    for i in range(12):
        ax3.text(idx[i], normal_cost[idx[i]] - 40, '%.4g' %normal_cost[idx[i]], fontsize=7, ha='center', va='center',zorder=12)
        ax3.text(idx[i], total_cost[idx[i]] + 30, '%.4g' %total_cost[idx[i]], fontsize=7, ha='center', va='center',zorder=12)
    ax3.grid(linestyle='--', axis='y', zorder=5)
    plt.subplots_adjust(bottom=0.2, wspace=0.2, hspace=0.2)

    plt.savefig("./figure/cell_10_mali_assignment_bar.tif", bbox_inches='tight', )
    plt.savefig("./figure/cell_10_mali_assignment_bar.jpg", bbox_inches='tight', )

if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plot_heat_map_visit_time()
    # plot_heat_map_aoi()
    # plot_aoi_vary_with_worker_number()
    plot_multi_mali_assignment_bar()
    # plot_learn_and_gamma()
    # pho_win_plot()
    # mali_plot()
    # plot_mean_ptp_std()
    # plot_heat_map_visit_time()
    print('ok')

    # plot_aoi_vary_with_worker_number()
    # plot_aoi_vary_with_sensor_number()
    # x = [i for i in range(10)]
    # y = [[j for j in range(i, i + 10)] for i in range(4)]
    # leg = ("1","2","3","4")
    # plot_1_4_curve_fig((x, x, x, x),
    #                    (y[0], y[1], y[2], y[3]),
    #                    ("1", "2", "3", "4"),
    #                    ("A", "B", "C", "D"),
    #                    ([0, 10], [0,10], [0, 10], [0, 10]),
    #                    ([0, 20], [0, 20], [0, 20], [0, 20]),
    #                    ("a", "b", "c", "d"),
    #                    (leg, leg, leg, leg),
    #                    "test")
