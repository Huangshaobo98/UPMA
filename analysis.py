import numpy as np
import pandas
import pandas as pd
from scipy.signal import savgol_filter
import os
analysis_dir = 'analysis/'

def std(data):
    return np.std(data)

def ptp(data):
    return np.ptp(data)

def mean(data):
    return np.mean(data)

def learn_rate_analysis():
    learn_rate = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.00005]
    files = ["save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_{}_gama_0.9_epd_0.99999/train/episode.csv"
             .format(str(l)) for l in learn_rate]

    data = []
    unfinish = []
    for file in files:
        t = pandas.read_csv(file)
        data.append(t['average real aoi'].to_numpy() / 5000)
        unfinish.append(t['slot number'].to_numpy() == 1394)

    gamma = [0.99, 0.95, 0.9, 0.75, 0.5, 0.3]
    files = ["save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_{}_epd_0.99999/train/episode.csv"
             .format(str(l)) for l in gamma]

    data_ = []
    unfinish_ = []
    for file in files:
        t = pandas.read_csv(file)
        data_.append(t['average real aoi'].to_numpy() / 5000)
        unfinish_.append(t['slot number'].to_numpy() == 1394)

    return data, unfinish, data_, unfinish_


def aoi_vary_with_worker_number():
    wkr = [50, 500, 5000, 10000]
    file_dir = ["save/t-drive_sen_5000_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
             .format(str(w)) for w in wkr]
    files = ['test/Test.csv', 'compare/Greedy.csv', 'compare/RR.csv', 'compare/CCPP.csv']

    aoi_data = []
    for dir in file_dir:
        aoi_wkr_data = []
        for f in files:
            file = dir + f
            t = pandas.read_csv(file)
            g = savgol_filter(t['sum real aoi'].to_numpy() / 5000, 50, 3)
            aoi_wkr_data.append(g)

        aoi_data.append(aoi_wkr_data)

    x = [i for i in range(1, 1395)]
    x_ = [x for _ in range(4)]

    return (x_, x_, x_, x_), tuple(aoi_data)

def aoi_vary_with_sensor_number():
    sensor = [500, 1000, 5000, 10000]
    file_dir = ["save/t-drive_sen_{}_wkr_50_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
             .format(str(s)) for s in sensor]
    files = ['test/Test.csv', 'compare/Greedy.csv', 'compare/RR.csv', 'compare/CCPP.csv']

    aoi_data = []
    for idx, dir in enumerate(file_dir):
        aoi_wkr_data = []
        for f in files:
            file = dir + f
            t = pandas.read_csv(file)
            g = savgol_filter(t['sum real aoi'].to_numpy() / sensor[idx], 50, 3)
            aoi_wkr_data.append(g)

        aoi_data.append(aoi_wkr_data)

    x = [i for i in range(1, 1395)]
    x_ = [x for _ in range(4)]

    return (x_, x_, x_, x_), tuple(aoi_data)

def global_average_aoi():
    file_dir = "save/t-drive_sen_5000_wkr_500_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
    files = ["test/slot_aoi.npy", "compare/slot_aoi_Greedy.npy", "compare/slot_aoi_RR.npy", "compare/slot_aoi_CCPP.npy"]
    dfs = []
    for f in files:
        path = file_dir + f
        data = np.load(path)
        t = [i for i in range(1, 11)]
        dfs.append(pd.DataFrame(data, index=t, columns=t))

    return dfs

def global_path_time():
    file_dir = "save/t-drive_sen_5000_wkr_50_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
    files = ["test/Test.csv", "compare/Greedy.csv", "compare/RR.csv", 'compare/CCPP.csv']
    dfs = []

    for f in files:
        df = pd.read_csv(file_dir + f)
        [x, y] = [df['uav position x'].to_numpy(), df['uav position y'].to_numpy()]
        data = np.zeros(shape=(10, 10), dtype=int)
        for x_, y_ in zip(x, y):
            data[x_, y_] += 1

        t = [i for i in range(1, 11)]
        dfs.append(pd.DataFrame(data, index=t, columns=t))

    return dfs

def mean_ptp_std():
    file_dir1 = ["save/t-drive_sen_5000_wkr_{}_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
                .format(str(i)) for i in [0, 50, 500, 5000, 10000]]

    file_dir2 = ["save/t-drive_sen_{}_wkr_50_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/"
                .format(str(i)) for i in [500, 1000, 5000, 10000]]

    file_dir = file_dir1 + file_dir2

    files = ["test/Test.csv", "compare/Greedy.csv", "compare/RR.csv", "compare/CCPP.csv"]

    meand = np.zeros(shape=(len(file_dir), 4))
    ptpd = np.zeros(shape=(len(file_dir), 4))
    stdd = np.zeros(shape=(len(file_dir), 4))

    sensnums = [5000, 5000, 5000, 5000, 5000, 500, 1000, 5000, 10000]

    for idx in range(len(file_dir)):
        for k in range(4):
            path = file_dir[idx] + files[k]
            aoi_data = pd.read_csv(path)['sum real aoi'].to_numpy() / sensnums[idx]
            meand[idx, k] = mean(aoi_data[100:])
            ptpd[idx, k] = ptp(aoi_data[100:])
            stdd[idx, k] = std(aoi_data[100:])

    pd_mean = pd.DataFrame(meand, index=['M=0', 'M = 50', 'M = 500', 'M = 5000', 'M = 10000', 'N = 500', 'N = 1000', 'N = 5000', 'N = 10000'],
                           columns=['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP'])
    pd_ptp = pd.DataFrame(ptpd, index=['M=0', 'M = 50', 'M = 500', 'M = 5000', 'M = 10000', 'N = 500', 'N = 1000', 'N = 5000', 'N = 10000'],
                           columns=['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP'])
    pd_std = pd.DataFrame(stdd, index=['M=0', 'M = 50', 'M = 500', 'M = 5000', 'M = 10000', 'N = 500', 'N = 1000', 'N = 5000', 'N = 10000'],
                           columns=['DRL-GAM', 'Greedy', 'Robin Round', 'CCPP'])
    return pd_mean, pd_ptp, pd_std

def pho_settings():
    path = "save/t-drive_sen_5000_wkr_5000_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/test/Test"
    res = []
    good_trust = []
    bad_trust = []
    for pho in [0.3, 0.5, 0.7, 0.9]:
        data = []
        gdata = []
        bdata = []
        for i in range(100):
            df = pd.read_csv(path + '_{}_pho_{}.csv'.format(i, pho))
            t = df['norm'].to_numpy()
            gt = df['good trust'].to_numpy()
            bt = df['bad trust'].to_numpy()
            if len(t) == 1394:
                data.append(t)
                gdata.append(gt)
                bdata.append(bt)
        res.append(np.average(np.array(data), axis=0))
        good_trust.append(np.average(np.array(gdata), axis=0))
        bad_trust.append(np.average(np.array(bdata), axis=0))
        # res.append(np.average(data, axis=0))
    return res,good_trust,bad_trust

def win_settings():
    path = "save/t-drive_sen_5000_wkr_100_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/test/Test"
    res = []
    for win in [5, 10, 15]:
        data = []
        for i in range(100):
            t = pd.read_csv(path + '_{}_win_{}.csv'.format(i, win))['norm'].to_numpy()
            if len(t) == 1394:
                data.append(t)

        res.append(np.average(np.array(data), axis=0))
        # res.append(np.average(data, axis=0))
    return res

def mali_compare():
    path = "save/t-drive_sen_5000_wkr_5000_epi_500_bat_256_lr_5e-05_gama_0.9_epd_0.99995/test/Test"
    test_assign = []
    random_norm = []
    test_norm = []
    random_assign = []
    for mali in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        data_assign = []
        data_norm = []
        for rpt in range(100):
            df = pd.read_csv(path + '_{}_mali_{}.csv'.format(rpt, mali))
            mali_assign = df['malicious assignment'].to_numpy()
            normal_assign = df['normal assignment'].to_numpy()
            norm = df['norm'].to_numpy()
            if len(mali_assign) == 1394:
                data_assign.append(np.average(np.sum(mali_assign) / np.sum(mali_assign + normal_assign)))
                data_norm.append(np.average(norm))
        test_assign.append(np.average(data_assign))
        test_norm.append(np.average(data_norm))

    for mali in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        data_assign = []
        data_norm = []
        for rpt in range(100):
            df = pd.read_csv(path + '_{}_mali_{}_random.csv'.format(rpt, mali))
            mali_assign = df['malicious assignment'].to_numpy()
            normal_assign = df['normal assignment'].to_numpy()
            norm = df['norm'].to_numpy()
            if len(mali_assign) == 1394:
                data_assign.append(np.average(np.sum(mali_assign) / np.sum(mali_assign + normal_assign)))
                data_norm.append(np.average(norm))
        random_assign.append(np.average(data_assign))
        random_norm.append(np.average(data_norm))
    return random_assign, random_norm, test_assign, test_norm



if __name__ == '__main__':
    # [data, unfinish] = learn_rate_analysis()
    # x_tuple, y_tuple = aoi_vary_with_worker_number()
    # a = 1

    # [m, p, s] = mean_ptp_std()
    [random_assign, random_norm, test_assign, test_norm] = mali_compare()

    t = (np.array(random_norm) - np.array(test_norm)) / np.array(random_norm)
    print(t)
    print(np.average(t))

    g = (np.array(random_assign) - np.array(test_assign)) / np.array(random_assign)
    print(g)
    print(np.average(g))