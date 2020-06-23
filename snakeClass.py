import os
import pygame
import argparse
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from random import randint, random
from keras.utils import to_categorical

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1/2
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 150   # neurons in the first layer
    params['second_layer_size'] = 150   # neurons in the second layer
    params['third_layer_size'] = 150    # neurons in the third layer
    params['episodes'] = 15
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['weights_path'] = 'weights/weights2.hdf5'
    params['load_weights'] = True
    params['train'] = True
    params['verbose'] = False
    return params


class Game:
    def __init__(self, game_width, game_height, display):
        self.display = display
        self.game_width = game_width
        self.game_height = game_height
        if display:
            self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
            pygame.display.set_caption('SnakeGen')
            self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):
    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.action = []
        if game.display:
            self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.crash = True
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food(object):
    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position:
            return self.x_food, self.y_food
        else:
            self.food_coord(game, player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 16)
    myfont_bold = pygame.font.SysFont('Segoe UI', 16, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    if len(game.player.action):
        text_predict = myfont.render('ACTION: ', True, (0, 0, 0))
        text_predict_number = myfont_bold.render(str(game.player.action), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    if len(game.player.action):
        game.gameDisplay.blit(text_predict, (45, 460))
        game.gameDisplay.blit(text_predict_number, (120, 460))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)


def update_screen():
    pygame.display.update()


def initialize_game(player, game, food, agent, batch_size):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food, agent)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(player, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="b",
        x_jitter=.1,
        line_kws={'color': 'green'}
    )
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def run(display_option, speed, params):
    if display_option:
        pygame.init()
    agent = DQNAgent(params)

    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    while counter_games < params['episodes']:
        # Initialize classes
        game = Game(440, 440, display_option)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent, params['batch_size'])
        if display_option:
            display(player1, food1, game, record)
        step_count = 0
        raw_data = [] if params['raw_output'] is not None else None
        while not game.crash:
            if display_option:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
            if not params['train']:
                agent.epsilon = 0
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, player1, food1)

            # perform random actions based on agent.epsilon, or choose the action
            if random() < agent.epsilon or random() < (step_count - 300) / 300:
                prediction = [0, 0, 0]
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1, agent)
            game.player.action = final_move
            state_new = agent.get_state(game, player1, food1)

            # set reward for the new state
            reward = agent.set_reward(player1, game.crash)
            if raw_data is not None:
                raw_data.append({
                    'head_x': player1.x,
                    'head_y': player1.y,
                    'food_x': food1.x_food,
                    'food_y': food1.y_food,
                    'snake_position': player1.position[:],
                    'snake_x_change': player1.x_change,
                    'snake_y_change': player1.y_change,
                    'eaten': player1.eaten,
                    'reward': reward,
                    'crash': game.crash
                })

            if params['verbose']:
                print(prediction, final_move, reward, step_count)

            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)
            step_count += 1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        print(f'Game {counter_games}/{step_count}/{agent.epsilon}      Score: {game.score}')
        if raw_data is not None:
            fn_index = params['raw_output_index'] + counter_games
            with open(os.path.join(params['raw_output'], f'{fn_index}.json'), 'w') as out:
                json.dump(raw_data, out)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    if params['train']:
        agent.model.save_weights(params['weights_path'])
    plot_seaborn(counter_plot, score_plot)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument('--display', type=str2bool, default=False)
    parser.add_argument('--no-gpu', type=str2bool, default=False)
    parser.add_argument('--speed', type=int, default=10)
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--raw-output', type=str, default=None)
    parser.add_argument('--raw-output-index', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--train', type=str2bool, default=True)

    args = parser.parse_args()
    params['bayesian_optimization'] = False    # Use bayesOpt.py for Bayesian Optimization
    params['verbose'] = args.verbose
    params['raw_output'] = args.raw_output
    params['raw_output_index'] = args.raw_output_index
    params['train'] = args.train
    params['episodes'] = args.episodes
    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.display:
        pygame.font.init()
    run(args.display, args.speed, params)
