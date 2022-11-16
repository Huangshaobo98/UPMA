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

# from time import time

# from keras.callbacks import TensorBoard
# from keras.utils.vis_utils import plot_model

EPISODES = 5000

class DQNAgent:
    def __init__(self, cell_size, action_size, gamma=0.9, epsilon=1, epsilon_decay=0.99999,
                 epsilon_min=0.08, lr=0.0005, dueling=True, continue_train=False, continue_train_path=""):
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
        if continue_train:
            self.load(continue_train_path)
        self.update_target_model()

    @staticmethod
    def huber_loss(y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def transform(self, pos_state):
        if not self.pos_grid:
            new_pos_state = np.zeros(shape=(2, self.cell_size,), dtype=np.float64)
            new_pos_state[0, pos_state[0]] = 1
            new_pos_state[1, pos_state[1]] = 1
        else:
            new_pos_state = np.zeros(shape=(self.cell_size, self.cell_size), dtype=np.float64)
            new_pos_state[pos_state[0]][pos_state[1]] = 1
        return new_pos_state

    def _build_model(self, dueling):
        # Neural Net for Deep-Q learning Model
        InputA = Input(shape=(self.cell_size, self.cell_size))  # 观测AoI状态
        if not self.pos_grid:
            InputB = Input(shape=(2, self.cell_size))                    # 无人机坐标点
        else:
            InputB = Input(shape=(self.cell_size, self.cell_size))
        InputC = Input(shape=(1,))      # 能量模型，一维数值
        InputD = Input(shape=(1,))      # 是否在充电1.0-true 0.0 false

        # x = Conv2D(6, (3, 3), padding='same', activation='linear')(InputA)
        x = Flatten()(InputA)
        # x = Dense(64, activation='linear')(x)
        # x = Dense(64, activation='linear')(x)
        x = Model(inputs=InputA, outputs=x)

        # y = Dense(8, activation='linear')(InputB)
        # y = Dense(64, activation='linear')(InputB)
        y = Flatten()(InputB)
        y = Model(inputs=InputB, outputs=y)

        z = Flatten()(InputC)
        z = Model(inputs=InputC, outputs=z)

        q = Flatten()(InputD)
        q = Model(inputs=InputD, outputs=z)

        combined = K.concatenate([x.output, y.output, z.output, q.output])

        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        o = Dense(1024, activation='relu')(combined)
        o = Dense(512, activation='relu')(o)
        o = Dense(256, activation='relu')(o)
        o = Dense(256, activation='relu')(o)
        o = Dense(128, activation='relu')(o)
        o = Dense(64, activation='relu')(o)
        # z = Dense(, activation='relu')(z)
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

    def memorize(self,
                 prev_observation_aoi, next_observation_aoi,
                 prev_real_aoi, next_real_aoi,
                 prev_position, next_position,
                 prev_energy, next_energy,
                 prev_charge_state, next_charge_state,
                 uav_action_index, reward, done):

        prev_pos_state = self.transform(prev_position)
        next_pos_state = self.transform(next_position)

        self.memory.append((prev_observation_aoi, next_observation_aoi,
                            prev_real_aoi, next_real_aoi,
                            prev_pos_state, next_pos_state,
                            prev_energy, next_energy,
                            prev_charge_state, next_charge_state,
                            uav_action_index, reward, done))

    # 废弃
    def deprecated_act(self, real_aoi_state, observation_aoi_state, position_state, energy_state, predict_by_real: bool):
        pos_state = self.transform(position_state)
        if predict_by_real:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict([real_aoi_state[np.newaxis, :, :],
                                             pos_state[np.newaxis, :, :],
                                             energy_state[np.newaxis, :]], batch_size=1, verbose=0)
        else:
            act_values = self.model.predict([observation_aoi_state[np.newaxis, :, :],
                                             pos_state[np.newaxis, :, :],
                                             energy_state[np.newaxis, :]], batch_size=1, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def act(self, aoi_state, position_state, energy_state, charge_state):
        pos_state = self.transform(position_state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([aoi_state[np.newaxis, :, :],
                                         pos_state[np.newaxis, :, :],
                                         energy_state[np.newaxis, :],
                                         charge_state[np.newaxis, :]], batch_size=1, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = np.array(random.sample(self.memory, batch_size), dtype=object)

        # self.memory.append((prev_observation_aoi, next_observation_aoi,
        #                     prev_real_aoi, next_real_aoi,
        #                     prev_pos_state, next_pos_state,
        #                     prev_energy, next_energy,
        #                     prev_charge_state, next_charge_state,
        #                     uav_action_index, reward, done))

        prev_real_aoi_states = np.stack(minibatch[:, 2])
        next_real_aoi_states = np.stack(minibatch[:, 3])

        prev_observation_aoi = np.stack(minibatch[:, 0])
        next_observation_aoi = np.stack(minibatch[:, 1])

        prev_position_states = np.stack(minibatch[:, 4])
        next_position_states = np.stack(minibatch[:, 5])

        prev_energy = np.stack(minibatch[:, 6])
        next_energy = np.stack(minibatch[:, 7])

        prev_charge_state = np.stack(minibatch[:, 8])
        next_charge_state = np.stack(minibatch[:, 9])

        done = np.stack(minibatch[:, 12])

        reward = np.stack(minibatch[:, 11])
        action = np.stack(minibatch[:, 10])

        next_targets = self.model.predict([next_real_aoi_states,
                                           next_position_states,
                                           next_energy,
                                           next_charge_state], batch_size=batch_size, verbose=0)

        targets = self.model.predict([prev_real_aoi_states,
                                      prev_position_states,
                                      prev_energy,
                                      prev_charge_state], batch_size=batch_size, verbose=0)

        targets[range(batch_size), action] = reward + self.gamma * np.amax(next_targets, axis=1).reshape(reward.shape)
        targets[done, action[done]] = reward[done]
        self.model.fit([prev_real_aoi_states, prev_position_states, prev_energy, prev_charge_state],
                       targets, epochs=1, batch_size=batch_size, verbose=0)

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
