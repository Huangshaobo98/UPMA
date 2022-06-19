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
import numpy.random.rand as rand

# from keras.callbacks import TensorBoard
# from keras.utils.vis_utils import plot_model

EPISODES = 5000


class DQNAgent:
    def __init__(self, cell_size, action_size, gamma=0.9, epsilon=1, epsilon_decay=0.999,
                 epsilon_min=0.01, lr=0.001, dueling=True):
        self.cell_size = cell_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 创建双端队列
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr

        self.model = self._build_model(dueling)
        self.target_model = self._build_model(dueling)  # 创建两个相同的网络模型
        self.update_target_model()

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def transform(self, pos_state):
        new_pos_state = np.zeros(shape=(2, self.cell_size,), dtype=np.float64)
        new_pos_state[0, pos_state[0]] = 1
        new_pos_state[1, pos_state[1]] = 1
        return new_pos_state

    def _build_model(self, dueling):
        # Neural Net for Deep-Q learning Model
        InputA = Input(shape=(self.cell_size, self.cell_size))
        InputB = Input(shape=(2, self.cell_size))

        # x = Conv2D(6, (3, 3), padding='same', activation='linear')(InputA)
        x = Flatten()(InputA)
        # x = Dense(64, activation='linear')(x)
        # x = Dense(64, activation='linear')(x)
        x = Model(inputs=InputA, outputs=x)

        # y = Dense(8, activation='linear')(InputB)
        # y = Dense(64, activation='linear')(InputB)
        y = Flatten()(InputB)
        y = Model(inputs=InputB, outputs=y)
        combined = K.concatenate([x.output, y.output])

        # model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        z = Dense(256, activation='relu')(combined)
        z = Dense(256, activation='relu')(z)
        z = Dense(128, activation='relu')(z)
        z = Dense(64, activation='relu')(z)
        # z = Dense(, activation='relu')(z)
        if dueling:
            z = Dense(self.action_size + 1, activation='linear')(z)
            z = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_size,))(z)
        else:
            z = Dense(self.action_size, activation='linear')(z)

        model = Model(inputs=[x.input, y.input], outputs=z)
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.learning_rate))
        # plot_model(model, to_file='Flatten.png', show_shapes=True)
        return model

    def update_target_model(self):  # 将估计模型的权重赋予目标模型
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, AoI_state, position_state, action, reward, next_AoI_state, next_position_state, done):
        new_pos_state = self.transform(position_state)
        new_next_pos_state = self.transform(next_position_state)
        self.memory.append((AoI_state, new_pos_state, action, reward, next_AoI_state, new_next_pos_state, done))

    def act(self, AoI_state, positions):
        if rand() <= self.epsilon:
            return random.randrange(self.action_size)
        new_pos_state = self.transform(positions)
        act_values = self.model.predict([AoI_state[np.newaxis, :, :], new_pos_state[np.newaxis, :, :]], batch_size=1)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch = np.array(random.sample(self.memory, batch_size))

        AoI_states = np.stack(minibatch[:, 0])
        position_states = np.stack(minibatch[:, 1])

        next_AoI_states = np.stack(minibatch[:, 4])
        next_position_states = np.stack(minibatch[:, 5])

        next_targets = self.model.predict([next_AoI_states, next_position_states], batch_size=batch_size)
        targets = self.model.predict([AoI_states, position_states], batch_size=batch_size)

        done = np.stack(minibatch[:, 6])
        reward = np.stack(minibatch[:, 3])
        action = np.stack(minibatch[:, 2])

        targets[done, action[done]] = reward[done]
        targets[range(batch_size), action] = reward + self.gamma * np.amax(next_targets, axis=1).reshape(reward.shape)

        self.model.fit([AoI_states, position_states], targets, epochs=1, batch_size=batch_size, verbose=0)

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
