# 定期备份zip的脚本
import os
import time
import heapq
def ignore_path():
    ignore_path = [
        'data/t-drive',
        'data/t-drive.zip',
        'data/__pycache__',
        'data/data_x_10_y_10/worker_sec_382.3780704306932.npy',
        'data/coordinate.npy',
        'backup',
        '*.out',
        '.git',
        '__pycache__',
        'figure',
        'running.csv',
        'Test.csv'
    ]
    exclude_param = ''.join([' --exclude=' + ign for ign in ignore_path])
    return exclude_param

def old_file_clean(max_number):
    paths = os.listdir('./backup')
    to_delete = []
    if len(paths) < max_number:
        return
    for path in paths:
        rel_path = './backup/' + path
        heapq.heappush(to_delete, (os.path.getctime(rel_path), rel_path))

    (_, path) = heapq.heappop(to_delete)
    os.remove(path)


def backup():
    for i in range(180):
        os.system('touch ./backup/tar_running_do_not_transmit_data.txt')
        current = time.strftime("%Y-%m-%d_%H_%M_%S",time.localtime())
        filename = './backup/backup_{}.tar.gz'.format(current)
        tar_command = 'tar -zcvf ' + filename + ignore_path() + ' .'
        os.system(tar_command)
        old_file_clean(6)
        if os.path.exists('./backup/tar_running_do_not_transmit_data.txt'):
            os.remove('./backup/tar_running_do_not_transmit_data.txt')
        time.sleep(3600 * 4)


if __name__ == '__main__':
    backup()
    print('process {}'.format(os.getpid()))