import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from analysis import *
import seaborn as sns

save_path = 'figure/'
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
    names = ['$\mu=0.001$','$\mu=0.0005$', '$\mu=0.0001$', '$\mu=0.00005$', '$\mu=0.00001$']
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
    names = ['$\\rho=0.3$','$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']

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
        axg.plot([i for i in range(1, 1395)], bad[idx], color=plt.get_cmap('Set1')(idx), linestyle='--', label= 'Malicious: ' + i)
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
    names = ['malicious: 10%','malicious: 30%', 'malicious: 50%', 'malicious: 70%']
    # axl.grid(linestyle='--')
    axl.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])-0.02, random_norm, width=0.04, color=plt.get_cmap('tab10')(0), label='Random')
    axl.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])+0.02, test_norm, width=0.04,color=plt.get_cmap('tab10')(1), label='GMTA')

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
    axg.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])-0.02, random_assign, width=0.04, color=plt.get_cmap('tab10')(0), label='Random')
    axg.bar(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])+0.02, test_assign, width=0.04, color=plt.get_cmap('tab10')(1), label='GMTA')
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
    titles = ['(a) SN number $N$ = 500', '(b) SN number $N$ = 1000', '(c) SN number $N$ = 5000', '(d) SN number $N$ = 10000']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(sensors), len(workers)), dtype=np.float64)
    compare_reduce_rate = []
    for i in range(4):
        for idw, w in enumerate(workers):
            for ids, s in enumerate(sensors):
                pathname = 'save/t-drive_sen_{}_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999'\
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
            ax.bar([i + diff[idx1] for i in range(len(workers))], matrix[idx1][idx, :], width=width, color=plt.get_cmap('tab20')(idx1), label=labels[idx1], zorder=10)

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
    titles = ['(a) SN number $N$ = 500', '(b) SN number $N$ = 1000', '(c) SN number $N$ = 5000', '(d) SN number $N$ = 10000']
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(sensors), len(workers)), dtype=np.float64)
    compare_reduce_rate = []
    for i in range(4):
        for idw, w in enumerate(workers):
            for ids, s in enumerate(sensors):
                pathname = 'save/t-drive_sen_{}_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999'\
                    .format(str(s), str(w)) + paths[i]
                npz = np.load(pathname)
                matrix[i, ids, idw] = np.average(np.std(npz['avg_real_aoi'][npz['actual_slot'] == 1394][:, 100:], axis=1))
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
            ax.bar([i + diff[idx1] for i in range(len(workers))], matrix[idx1][idx, :], width=width, color=plt.get_cmap('tab20')(idx1), label=labels[idx1], zorder=5)

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
    titles = ['(a) Worker number $M$ = 100', '(b) Worker number $M$ = 500', '(c) Worker number $M$ = 1000', '(d) Worker number $M$ = 5000']
    upper_limits = [60, 54, 48, 18]
    step = [10, 9, 8, 3]
    paths = ['/test/Test.npz', '/compare/Greedy.npz', '/compare/RR.npz', '/compare/CCPP.npz']
    matrix = np.empty(shape=(4, len(workers), 1394), dtype=np.float64)
    for i in range(4):
        for idw, w in enumerate(workers):
            pathname = 'save/t-drive_sen_5000_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999'\
                .format(str(w)) + paths[i]
            npz = np.load(pathname)
            matrix[i, idw] = np.mean(npz['avg_real_aoi'][npz['actual_slot'] == 1394], axis=0)


    fig = plt.figure(figsize=(18, 4), dpi=300)

    axs = []
    for i in range(141, 145):
        axs.append(fig.add_subplot(i))

    for idx, ax in enumerate(axs):
        for idx1 in range(4):
            ax.plot([i for i in range(1, 1395)], matrix[idx1, idx], color=plt.get_cmap('tab10')(idx1), label=labels[idx1])

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
            'save/t-drive_sen_5000_wkr_5000_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_reduce_{}.npz'.format(str(reduce)))
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
                    annot_kws={'fontsize': 4}, cbar_kws={'label':'Average cell AoI (${\\Delta t}$)'})
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
    ticks = [[0, .5, 1, 1.5, 2, 2.5, 3], [0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]]
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
                           cbar_kws={'label':'Access probability of the UAV (%)', 'ticks': ticks[idx]})
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
            pathname = 'save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99999/test/Test_pho_{}_mali_{}_new.npz'\
                .format(str(p), str(m))
            npz = np.load(pathname)
            pho_datas[i, j] = np.mean(npz['norm'][npz['actual_slot'] == 1394][:, 200:])
            # pho_datas.append(pd.DataFrame(temp / np.sum(temp) * 100, index=t, columns=t))
            if m == 0.5:
                good_datas[i] = np.mean(npz['good_trust'][npz['actual_slot'] == 1394], axis=0)
                bad_datas[i] = np.mean(npz['bad_trust'][npz['actual_slot'] == 1394], axis=0)


    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\\rho=0.3$','$\\rho=0.5$', '$\\rho=0.7$', '$\\rho=0.9$']

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
        axg.plot([i for i in range(1, 1395)], good_datas[idx], color=plt.get_cmap('Set1')(idx), label='{}: normal'.format(i) )
        axg.plot([i for i in range(1, 1395)], bad_datas[idx], color=plt.get_cmap('Set1')(idx), linestyle='--', label= '{}: malicious '.format(i))
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
    titles = ['(a) Error performance', '(b) AoI performance', '(c) Tasks assigned to normal workers', '(d) Tasks assigned to malicious workers']
    x_label = 'Ratio of malicious workers'
    y_labels = ['2-Norm error of estimated and actual AoI', 'Average global AoI', 'The number of assigned tasks ($\\times10^3$)', 'The number of assigned tasks ($\\times10^3$)']
    for idx, ax in enumerate(axs):
        for idx1 in range(2):
            ax.bar(np.array(mali) + diff[idx1], matrix[idx][idx1], width=0.04, color=plt.get_cmap('tab10')(idx1), label=labels[idx1], zorder=5)

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

if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plot_heat_map_visit_time()
    # plot_heat_map_aoi()
    # plot_aoi_vary_with_worker_number()
    new_mali_plot()
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






