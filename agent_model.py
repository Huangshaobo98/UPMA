# -*- coding: utf-8 -*-
import random
# import gym
import numpy as np
from collections import deque
# from keras.models import Sequential

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

# EPISODES = 5000


class State:
    cell_size = g.cell_limit
    max_energy = g.uav_energy
    onehot_position = g.onehot_position
    sensor_number = g.sensor_number

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

    def __str__(self):
        msg = "Position: {}, charge state: {}, energy left: {}.\r\nreal aoi:\r\n{}\r\nobservation aoi:\r\n{}"\
            .format(self.position, self.charge,
                    self.energy, str(np.around(self.real_aoi, 0)), str(np.around(self.observation_aoi, 2)))
        return msg

    @property
    def real_aoi(self) -> np.ndarray:
        return self.__real_aoi

    @property
    def real_aoi_state(self) -> np.ndarray:
        return self.real_aoi / State.sensor_number

    @property
    def observation_aoi(self) -> np.ndarray:
        return self.__observation_aoi

    @property
    def observation_aoi_state(self) -> np.ndarray:
        return self.__observation_aoi / State.sensor_number

    @property
    def energy(self) -> float:
        return self.__energy

    @property
    def energy_state(self) -> np.ndarray:
        return np.array(self.energy / State.max_energy, dtype=np.float64)

    @property
    def charge(self) -> bool:
        return self.__charge

    @property
    def charge_state(self) -> np.ndarray:
        return np.array([1.0 if self.charge else 0.0])

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
            new_pos_state[0, position_state[0]] = 1
            new_pos_state[1, position_state[1]] = 1
        else:
            new_pos_state = np.zeros(shape=(State.cell_size, State.cell_size), dtype=np.float64)
            new_pos_state[position_state[0]][position_state[1]] = 1
        return new_pos_state

    def pack_observation(self) -> List[np.ndarray]:
        return [self.observation_aoi[np.newaxis, :, :], self.position_state[np.newaxis, :, :],
                self.energy_state[np.newaxis, :, :], self.charge_state[np.newaxis, :, :]]

    def pack_real(self) -> List[np.ndarray]:
        return [self.real_aoi[np.newaxis, :, :], self.position_state[np.newaxis, :, :],
                self.energy_state[np.newaxis, :, :], self.charge_state[np.newaxis, :, :]]


class DQNAgent:
    def __init__(self, cell_size, action_size, gamma=0.9, epsilon=1, epsilon_decay=0.99999,
                 epsilon_min=0.08, lr=0.0005, dueling=True, train=True, continue_train=False, model_path=""):
        # 暂且设定的动作集合：'h': 六个方向的单元移动+一种什么都不做的悬浮，在特定小区的悬浮可以看做是进行了充电操作
        self.cell_size = cell_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 创建双端队列
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.pos_grid = False
        self.model = self._build_model(dueling)
        self.target_model = self._build_model(dueling)  # 创建两个相同的网络模型
        self.train = train
        if continue_train:
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
            return random.randrange(self.action_size)
        if self.train:
            act_values = self.model.predict(state.pack_real(), batch_size=1, verbose=0)
        else:
            act_values = self.model.predict(state.pack_observation(), batch_size=1, verbose=0)
        return np.argmax(act_values[0])  # returns action

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
            prev_real_aoi_states.append(prev_state.real_aoi)
            prev_position_states.append(prev_state.position_state)
            prev_energies.append(prev_state.energy_state)
            prev_charge_states.append(prev_state.charge_state)

            next_real_aoi_states.append(next_state.real_aoi)
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

        # prev_real_aoi_states = np.stack(minibatch[:, 2])
        # next_real_aoi_states = np.stack(minibatch[:, 3])
        #
        # prev_observation_aoi = np.stack(minibatch[:, 0])
        # next_observation_aoi = np.stack(minibatch[:, 1])
        #
        # prev_position_states = np.stack(minibatch[:, 4])
        # next_position_states = np.stack(minibatch[:, 5])
        #
        # prev_energy = np.stack(minibatch[:, 6])
        # next_energy = np.stack(minibatch[:, 7])
        #
        # prev_charge_state = np.stack(minibatch[:, 8])
        # next_charge_state = np.stack(minibatch[:, 9])
        #
        # done = np.stack(minibatch[:, 12])
        #
        # reward = np.stack(minibatch[:, 11])
        # action = np.stack(minibatch[:, 10])

        next_targets = self.model.predict(next_state_stack, batch_size=batch_size, verbose=0)

        targets = self.model.predict(prev_state_stack, batch_size=batch_size, verbose=0)

        targets[range(batch_size), action_stack] = \
            reward_stack + self.gamma * np.amax(next_targets, axis=1).reshape(reward_stack.shape)
        targets[done_stack, action_stack[done_stack]] = reward_stack[done_stack]  # 这里改了一下位置，done状态下的target等价于其reward
        self.model.fit(prev_state_stack, targets, epochs=1, batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# if __name__ == "__main__":
#     env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-ddqn.h5")
#     done = False
#     batch_size = 32
#
#     for e in range(EPISODES):
#         # timess = time()
#         state = env.reset()
#         # print('reset: {}'.format(time()-timess))
#         state = np.reshape(state, [1, state_size])
#         for timeee in range(500):
#             env.render()
#             timess = time()
#             action = agent.act(state)
#             # print('act: {}'.format(time() - timess))
#             timess = time()
#             next_state, reward, done, _ = env.step(action)
#             # print('step: {}'.format(time() - timess))
#             # reward = reward if not done else -10
#             x, x_dot, theta, theta_dot = next_state
#             r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#             r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#             reward = r1 + r2
#
#             next_state = np.reshape(next_state, [1, state_size])
#             timess = time()
#             agent.memorize(state, action, reward, next_state, done)
#             # print('memorize: {}'.format(time() - timess))
#             state = next_state
#             if done:
#                 timess = time()
#                 agent.update_target_model()
#                 # print('update: {}'.format(time() - timess))
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, EPISODES, timeee, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 timess = time()
#                 agent.replay(batch_size)
#                 # print('replay: {}'.format(time() - timess))
#         # if e % 10 == 0:
#         #     agent.save("./save/cartpole-ddqn.h5")
