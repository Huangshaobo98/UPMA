import numpy as np

class Compare:
    def __init__(self,
                 x_limit,
                 y_limit,
                 method,
                 max_energy):
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.max_energy = max_energy
        self.method = method
        self.actions_1 = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 0], [0, 0]]
        self.actions_0 = [[-1, -1], [0, -1], [1, 0], [0, 1], [-1, 1], [-1, 0], [0, 0]]
        self.cell_visited = np.zeros(shape=(self.x_limit, self.y_limit), dtype=bool)

    def run(self, prev_state, move_energy):
        [cx, cy] = prev_state.position
        energy_left = prev_state.energy
        energy_consume = move_energy
        obv_aoi = prev_state.observation_aoi_state
        if self.method == "RR":
            return self.RoundRobin(cx, cy, energy_left, energy_consume)
        elif self.method == "Greedy":
            return self.Greedy(cx, cy, obv_aoi, energy_left, energy_consume)
        elif self.method == "CCPP":
            return self.CCPP(cx, cy, obv_aoi, energy_left, energy_consume)
        else:
            assert False

    def RoundRobin(self, cx, cy, energy_left, energy_consume):
        if (energy_left - energy_consume) / self.max_energy < 0.2:
            return 6

        if cx == 0:
            if cy == 0:
                return 2 # 起始状态移动
            else:
                if cy % 2 == 0:
                    return 1 # 返回初始位置
                else:
                    return 0

        if cy % 2 == 0:
            if cx == self.x_limit - 1:
                return 3
            else:
                return 2
        else:
            if cx == 1:
                if cy == self.y_limit - 1:
                    return 5
                return 4
            else:
                return 5

    def Greedy(self, cx, cy, obv_aoi, energy_left, energy_consume):
        if (energy_left - energy_consume) / self.max_energy < 0.2:
            return 6
        # if random.random() < 0.05:
        #     return random.randint(0, 5)
        max_aoi = 0
        index = -1
        if cy % 2 == 0:
            for idx, act in enumerate(self.actions_0):
                [nx, ny] = [cx + act[0], cy + act[1]]
                if nx >= 0 and nx < self.x_limit and ny >= 0 and ny < self.y_limit:
                    if obv_aoi[nx, ny] > max_aoi:
                        max_aoi = obv_aoi[nx, ny]
                        index = idx
        else:
            for idx, act in enumerate(self.actions_1):
                [nx, ny] = [cx + act[0], cy + act[1]]
                if nx >= 0 and nx < self.x_limit and ny >= 0 and ny < self.y_limit:
                    if obv_aoi[nx, ny] > max_aoi:
                        max_aoi = obv_aoi[nx, ny]
                        index = idx
        return index

    def get_path(self, cx, cy, obv_aoi):
        # actions = np.array([(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 0)])
        cur_pos = [cx, cy]
        path = np.empty(shape=(self.x_limit, self.y_limit), dtype=object)
        path[tuple(cur_pos)] = np.array([], dtype=int)  #之前的代码是直接使用位置来写的，这里我觉得可以改成使用动作
        path_AoI = np.zeros(shape=(self.x_limit, self.y_limit))
        cover_state = np.zeros(shape=(self.x_limit, self.y_limit))
        cover_state[tuple(cur_pos)] = 1
        layer_cells = np.array([cur_pos])
        layers = np.empty(shape=(2 * max(self.x_limit, self.y_limit) - 2,), dtype=object)
        count = 0
        while not cover_state.all():
            new_layer_cells = np.empty(shape=(0, 2), dtype=np.int32)
            for cell in layer_cells:
                if cell[1] % 2 == 0:
                    for act_idx, action in enumerate(self.actions_0):
                        temp_pos = action + cell
                        if 0 <= temp_pos[0] < self.x_limit and 0 <= temp_pos[1] < self.y_limit and cover_state[tuple(temp_pos)] == 0:
                            if path_AoI[tuple(temp_pos)] < path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1:
                                path_AoI[tuple(temp_pos)] = path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1
                                path[tuple(temp_pos)] = np.append(path[tuple(cell)], act_idx)
                                new_layer_cells = np.vstack((new_layer_cells, temp_pos))
                else:
                    for act_idx, action in enumerate(self.actions_1):
                        temp_pos = action + cell
                        if 0 <= temp_pos[0] < self.x_limit and 0 <= temp_pos[1] < self.y_limit and cover_state[tuple(temp_pos)] == 0:
                            if path_AoI[tuple(temp_pos)] < path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1:
                                path_AoI[tuple(temp_pos)] = path_AoI[tuple(cell)] + obv_aoi[tuple(temp_pos)] + 1
                                path[tuple(temp_pos)] = np.append(path[tuple(cell)], act_idx)
                                new_layer_cells = np.vstack((new_layer_cells, temp_pos))
            layer_cells = np.unique(new_layer_cells, axis=0)
            for cell in layer_cells:
                cover_state[tuple(cell)] = 1
            layers[count] = layer_cells
            count += 1
        return path, path_AoI, layers

    def CCPP(self, cx, cy, obv_aoi, energy_left, energy_consume):
        if (energy_left - energy_consume) / self.max_energy < 0.2:
            return 6
        path, path_AoI, layers = self.get_path(cx, cy, obv_aoi)
        self.cell_visited[cx, cy] = True
        for i in range(self.x_limit):
            for j in range(self.y_limit):
                if obv_aoi[i, j] <= 0.00001:
                    self.cell_visited[i, j] = True
        if self.cell_visited.all():
            self.cell_visited = np.zeros(shape=(self.x_limit, self.y_limit), dtype=bool)
            self.cell_visited[cx, cy] = True
        max_val = 0
        max_path_act = np.empty(shape=(0, 2), dtype=np.int32)
        for layer in layers:
            for cell in layer:
                if self.cell_visited[tuple(cell)] == 0:
                    if path_AoI[tuple(cell)] > max_val:
                        max_val = path_AoI[tuple(cell)]
                        max_path_act = path[tuple(cell)][0]
            if max_val > 0:
                break
        return max_path_act