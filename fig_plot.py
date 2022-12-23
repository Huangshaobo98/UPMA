import os.path

import matplotlib.pyplot as plt
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
    save_name = save_path + 'learn_rate_gamma'
    [lr_data, lr_unfinish, gamma_data, gamma_unfinish] = learn_rate_analysis()
    fig = plt.figure(figsize=(9, 4), dpi=300)
    axl = fig.add_subplot(121)
    names = ['$\mu=0.005$','$\mu=0.001$', '$\mu=0.0005$', '$\mu=0.0001$', '$\mu=0.00005$']
    for idx, i in enumerate(names):
        axl.plot([i for i in range(1, 501)], lr_data[idx], color=plt.get_cmap('tab10')(idx), label=i)
        print(idx)
    axl.set_xlabel('Episodes')
    axl.set_ylabel('Global AoI')
    axl.set_xlim(0, 250)
    axl.set_ylim(0, 300)
    axl.legend()
    axl.set_title('(a) Learning rate $\mu$', y=-0.26)
    axl.grid(linestyle='--')
    axg = fig.add_subplot(122)

    names = ['$\gamma=0.95$', '$\gamma=0.9$', '$\gamma=0.75$']
    for idx, i in enumerate(names):
        axg.plot([i for i in range(1, 301)], gamma_data[idx][0:300], color=plt.get_cmap('Set1')(idx), label=i)
        print(idx)
    axg.set_xlabel('Episodes')
    axg.set_ylabel('Global AoI')
    axg.set_xlim(0, 250)
    axg.set_ylim(0, 300)
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
    axl.set_ylabel('2-Norm difference of estimated and actual AoI')
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
    axl.set_ylabel('2-Norm difference of estimated and actual AoI')
    # axl.set_xlim(0.05)
    # axl.set_ylim(0, 300)
    axl.legend()
    axl.set_title('(a) Difference of estimated and actual AoI', y=-0.26)
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

if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plot_heat_map_visit_time()
    # plot_heat_map_aoi()
    # plot_aoi_vary_with_worker_number()
    plot_aoi_vary_with_sensor_number()
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






