# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.layers.core.dense import Dense
from keras.layers.core.lambda_layer import Lambda
from keras.layers.reshaping.flatten import Flatten
from keras.engine.input_layer import Input
from keras.layers.regularization.dropout import Dropout
from keras.layers.pooling.max_pooling2d import MaxPool2D
from keras.optimizers.optimizer_v2.adam import Adam
from keras.engine.training import Model
import keras.backend as K
from typing import List
# from time import time
from global_parameter import Global as g
# from keras.callbacks import TensorBoard
# from keras.utils.vis_utils import plot_model
from math import floor
# EPISODES = 5000


class State:
    cell_size = 0
    max_energy = g.uav_energy
    onehot_position = g.onehot_position
    sensor_number = 0
    second_per_slot = g.sec_per_slot

    def __init__(self,
                 real_aoi: np.ndarray,
                 observation_aoi: np.ndarray,
                 position: List[int],
                 energy: float,
                 charge: bool,
                 ):

        self.__real_aoi = real_aoi
        self.__observation_aoi = observation_aoi
        # 注意，单纯使用self.position返回的是类型原有格式，使用self.position_state返回的是类型经过转换后的格式
        self.__position = position
        self.__energy = energy
        self.__charge = charge

    @staticmethod
    def init(sensor_number, cell_size):
        State.sensor_number = sensor_number
        State.cell_size = cell_size

    @property
    def real_aoi(self) -> np.ndarray:
        return self.__real_aoi

    @property
    def real_aoi_state(self) -> np.ndarray:
        return self.real_aoi / State.sensor_number #/ State.second_per_slot  #

    @property
    def average_real_aoi_state(self) -> float:
        return np.average(self.real_aoi_state)

    @property
    def observation_aoi(self) -> np.ndarray:
        return self.__observation_aoi

    @property
    def observation_aoi_state(self) -> np.ndarray:
        return self.__observation_aoi / State.sensor_number  # / State.second_per_slot  #

    @property
    def average_observation_aoi_state(self) -> float:
        return np.average(self.observation_aoi_state)

    @property
    def energy(self) -> float:
        return self.__energy

    @property
    def energy_state(self) -> np.ndarray:
        return np.array([self.energy / State.max_energy], dtype=np.float64)

    @property
    def charge(self) -> bool:
        return self.__charge

    @property
    def charge_state(self) -> np.ndarray:
        return np.array([1.0 if self.charge else 0.0], dtype=np.float64)

    @property
    def position(self) -> List[int]:
        return self.__position

    @property
    def position_state(self) -> np.ndarray:
        return self.transform(self.position)

    @staticmethod
    def transform(position_state: List[int]) -> np.ndarray:
        if State.onehot_position:
            new_pos_state = np.zeros(shape=(2, State.cell_size), dtype=np.float64)
            new_pos_state[0, position_state[0]] = 1.0
            new_pos_state[1, position_state[1]] = 1.0
        else:
            new_pos_state = np.zeros(shape=(State.cell_size, State.cell_size), dtype=np.float64)
            new_pos_state[position_state[0]][position_state[1]] = 1.0
        return new_pos_state

    @property
    def pack_observation(self) -> List[np.ndarray]:
        return [self.observation_aoi_state[np.newaxis, :, :], self.position_state[np.newaxis, :, :],
                self.energy_state[np.newaxis, :], self.charge_state[np.newaxis, :]]

    @property
    def pack_real(self) -> List[np.ndarray]:
        return [self.real_aoi_state[np.newaxis, :, :], self.position_state[np.newaxis, :, :],
                self.energy_state[np.newaxis, :], self.charge_state[np.newaxis, :]]


class DQNAgent:
    def __init__(self, cell_size, action_size, gamma=0.75, epsilon=1, epsilon_decay=0.99995,
                 epsilon_min=0.05, lr=0.0005, dueling=True, train=True, continue_train=False, model_path=""):
        # 暂且设定的动作集合：'h': 六个方向的单元移动+一种什么都不做的悬浮，在特定小区的悬浮可以看做是进行了充电操作
        self.cell_size = cell_size
        self.action_size = action_size
        self.memory = deque(maxlen=25000)  # 创建双端队列
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.pos_grid = False
        self.model = self._build_model(dueling)
        self.target_model = self._build_model(dueling)  # 创建两个相同的网络模型
        self.train = train
        if continue_train or not train:
            self.load(model_path)
        self.update_target_model()

    @staticmethod
    def huber_loss(y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def _build_model(self, dueling):
        # Neural Net for Deep-Q learning Model
        input_a = Input(shape=(self.cell_size, self.cell_size))  # 观测AoI状态
        if not self.pos_grid:
            input_b = Input(shape=(2, self.cell_size))                    # 无人机坐标点
        else:
            input_b = Input(shape=(self.cell_size, self.cell_size))
        input_c = Input(shape=(1,))      # 能量模型，一维数值
        input_d = Input(shape=(1,))      # 是否在充电1.0-true 0.0 false

        # x = Conv2D(6, (3, 3), padding='same', activation='linear')(InputA)
        x = Flatten()(input_a)
        # x = Dense(64, activation='linear')(x)
        # x = Dense(64, activation='linear')(x)
        x = Model(inputs=input_a, outputs=x)

        # y = Dense(8, activation='linear')(InputB)
        # y = Dense(64, activation='linear')(InputB)
        y = Flatten()(input_b)
        y = Model(inputs=input_b, outputs=y)

        z = Flatten()(input_c)
        z = Model(inputs=input_c, outputs=z)

        q = Flatten()(input_d)
        q = Model(inputs=input_d, outputs=q)

        combined = K.concatenate([x.output, y.output, z.output, q.output])

        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        o = Dense(256, activation='relu')(combined)
        o = Dense(256, activation='relu')(o)
        o = Dense(64, activation='relu')(o)
        # o = Dense(64, activation='relu')(o)
        # o = Dense(128, activation='relu')(o)
        o = Dense(64, activation='relu')(o)
        if dueling:
            o = Dense(self.action_size + 1, activation='linear')(o)
            o = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                       output_shape=(self.action_size,))(o)
        else:
            o = Dense(self.action_size, activation='linear')(o)

        model = Model(inputs=[x.input, y.input, z.input, q.input], outputs=o)
        model.compile(loss=self.huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        # plot_model(model, to_file='Flatten.png', show_shapes=True)
        return model

    def update_target_model(self):  # 将估计模型的权重赋予目标模型
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, prev_state: State, action: int, next_state: State, reward: float, done: bool):

        # prev_pos_state = self.transform(prev_state.position_state)
        # next_pos_state = self.transform(next_state.position_state)
        self.memory.append((prev_state, action, next_state, reward, done))

    def act(self, state: State):
        if self.train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), []
        if self.train:
            act_values = self.model.predict(state.pack_real, batch_size=1, verbose=0)
        else:
            act_values = self.model.predict(state.pack_observation, batch_size=1, verbose=0)
        return np.argmax(act_values[0]), list(act_values[0])  # returns action

    @staticmethod
    def __batch_stack(minibatch):
        prev_real_aoi_states = []
        prev_position_states = []
        prev_energies = []
        prev_charge_states = []

        next_real_aoi_states = []
        next_position_states = []
        next_energies = []
        next_charge_states = []

        actions = []
        rewards = []
        dones = []

        for prev_state, action, next_state, reward, done in minibatch:
            prev_real_aoi_states.append(prev_state.real_aoi_state)
            prev_position_states.append(prev_state.position_state)
            prev_energies.append(prev_state.energy_state)
            prev_charge_states.append(prev_state.charge_state)

            next_real_aoi_states.append(next_state.real_aoi_state)
            next_position_states.append(next_state.position_state)
            next_energies.append(next_state.energy_state)
            next_charge_states.append(next_state.charge_state)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        prev_real_aoi_stack = np.stack(prev_real_aoi_states)
        prev_position_stack = np.stack(prev_position_states)
        prev_energy_stack = np.stack(prev_energies)
        prev_charge_stack = np.stack(prev_charge_states)
        prev_state_stack_list = [prev_real_aoi_stack, prev_position_stack, prev_energy_stack, prev_charge_stack]

        next_real_aoi_stack = np.stack(next_real_aoi_states)
        next_position_stack = np.stack(next_position_states)
        next_energy_stack = np.stack(next_energies)
        next_charge_stack = np.stack(next_charge_states)
        next_state_stack_list = [next_real_aoi_stack, next_position_stack, next_energy_stack, next_charge_stack]

        action_stack = np.stack(actions)
        reward_stack = np.stack(rewards)
        done_stack = np.stack(dones)

        return prev_state_stack_list, next_state_stack_list, action_stack, reward_stack, done_stack

    def replay(self, batch_size):

        minibatch = np.array(random.sample(self.memory, batch_size), dtype=object)
        prev_state_stack, next_state_stack, action_stack, reward_stack, done_stack = self.__batch_stack(minibatch)

        next_targets = self.model.predict(next_state_stack, batch_size=batch_size, verbose=0)

        targets = self.model.predict(prev_state_stack, batch_size=batch_size, verbose=0)

        targets[range(batch_size), action_stack] = reward_stack + self.gamma * np.amax(next_targets, axis=1)
        targets[done_stack, action_stack[done_stack]] = reward_stack[done_stack]  # 这里改了一下位置，done状态下的target等价于其reward

        self.model.fit(prev_state_stack, targets, epochs=1, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
