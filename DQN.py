from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import random
import os
import numpy as np
import pandas as pd
from operator import add
import collections

class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        # input shape: 3 images, 22x22 map, 3 layers (22, 22, 9):
        # layer 1: walls of first image
        # layer 2: snake of first image
        # layer 3: food of first image
        # layer 4: walls of second image
        # layer 5: snake of second image
        # layer 6: food of second image
        # layer 7: walls of thrid image
        # layer 8: snake of thrid image
        # layer 9: food of thrid image
        model.add(Conv2D(filters=512, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(22, 22, 9)))
        model.add(Conv2D(filters=512, kernel_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='sigmoid'))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='sigmoid'))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights and os.path.isfile(self.weights):
            model.load_weights(self.weights)
        return model

    def get_state(self, game, player, food, previous_state = None):
        # try to build (22, 22, 9) array.
        state = []
        for y in range(22):
            state.append([])
            for x in range(22):
                state[y].append([
                    previous_state[y][x][3] if previous_state is not None else 0, # state - 2 wall
                    previous_state[y][x][4] if previous_state is not None else 0, # state - 2 snake
                    previous_state[y][x][5] if previous_state is not None else 0, # state - 2 food
                    previous_state[y][x][6] if previous_state is not None else 0, # state - 1 wall
                    previous_state[y][x][7] if previous_state is not None else 0, # state - 1 wall
                    previous_state[y][x][8] if previous_state is not None else 0, # state - 1 wall
                ])
                # wall
                state[y][x].append(1 if x == 0 or y == 0 or x == 21 or y == 21 else 0)
                # snake
                if player.x / 20 == x and player.y / 20 == y:
                    state[y][x].append(1)
                elif [x, y] in player.position:
                    state[y][x].append(0.5)
                else:
                    state[y][x].append(0)
                # food
                state[y][x].append(1 if x == food.x_food / 20 and y == food.y_food / 20 else 0)

        return np.array(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done, verbose=0):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))
        target_f[0][np.argmax(action)] = target
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=verbose)
