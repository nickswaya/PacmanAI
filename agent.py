from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pauser
from levels import LevelController
from text import TextGroup
from sprites import Spritesheet
from maze import Maze
from vector import Vector2
from constants import *
from pygame.locals import *
from entity import MazeRunner
from animation import Animation
import pygame
from stack import Stack
from model import Linear_QNet, QTrainer
from helper import *
import torch
import random
import numpy as np
from collections import deque
from run import GameController


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(10, 256, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        pacman_position = game.pacman.position.asInt()
        # pacman_direction = game.pacman.direction.asInt()
        ghost_positions = [ghost.position.asInt() for ghost in game.ghosts]
        # ghost_directions = [ghost.direction.asInt() for ghost in game.ghosts]
        # state = [pacman_position, pacman_direction, ghost_positions, ghost_directions]
        state = [pacman_position, ghost_positions]

        state = add_flatten_lists(state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 1500 - self.n_games
        # up, down, left, right
        final_move = [0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 4)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


action = [0,0,0,0,0]

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = GameController()
    game.startGame()
    agent = Agent()
    while True:
        game.update(action)
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)

        reward, done, score = game.update(final_move)
        # print(reward)
        # perform move and get new state
        state_new = agent.get_state(game)

        # # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.startGame()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':       
    train()