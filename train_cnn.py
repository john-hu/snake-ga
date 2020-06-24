import argparse
import json
import numpy as np
import os

from DQN import DQNAgent
from snakeClass import define_parameters

class FakePlayer:
  def __init__(self, x, y, position):
    self.x = x
    self.y = y
    self.position = position

class FakeFood:
  def __init__(self, x, y):
    self.x_food = x
    self.y_food = y


def fake_it(step):
  fake_player = FakePlayer(step['head_x'], step['head_y'], step['snake_position'])
  fake_food = FakeFood(step['food_x'], step['food_y'])
  return (fake_player, fake_food)

# train the model with the traditional cnn: prepare all states and actions
def train_cnn(args, agent, files):
  fn_count = 0
  states = []
  actions = []
  for fn in files:
    with open(os.path.join(args.raw_output, fn)) as json_file:
      steps = json.load(json_file)
    for step in steps:
      (fake_player, fake_food) = fake_it(step)
      states.append(agent.get_state(None, fake_player, fake_food))
      actions.append(step['action'])
    fn_count += 1
    print(f'{fn} loaded: {len(actions)} from {fn_count} / {len(files)}')
  print('training')
  agent.model.fit(np.array(states), np.array(actions), epochs=1, verbose=1, batch_size=100, shuffle=True)

# train the model with the qlearning way, create state and next_state and train
# (use the agent to train it)
def train_qlearning(args, agent, files):
  fn_count = 0
  for fn in files:
    with open(os.path.join(args.raw_output, fn)) as json_file:
      steps = json.load(json_file)
    print(f'training file {fn}: {fn_count} / {len(files)}')
    for step in steps:
      # prepare current state
      (fake_player, fake_food) = fake_it(step)
      state = agent.get_state(None, fake_player, fake_food)
      # prepare next state
      (fake_player_next, fake_food_next) = fake_it(step['next_state'])
      next_state = agent.get_state(None, fake_player_next, fake_food_next, state)
      # train with action, reward and crash
      agent.train_short_memory(state, step['action'], step['reward'], next_state, step['crash'], 2)
    fn_count += 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw-output", type=str, required=True)
  args = parser.parse_args()
  # prepare model
  params = define_parameters()
  params['train'] = True
  agent = DQNAgent(params)
  # list files
  if not os.path.isdir(args.raw_output):
    print(f'raw folder, {args.raw_output} is not a valid folder')
    return
  files = os.listdir(args.raw_output)
  files.sort()
  files = list(filter(lambda fn: fn.lower()[-5:] == '.json', files))
  print(f'training files count: {len(files)}')
  # training
  train_cnn(args, agent, files)
  # train_qlearning(args, agent, files)

  # save the result
  agent.model.save_weights(params['weights_path'])


if __name__ == "__main__":
    main()
