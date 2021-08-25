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
from vector import Vector2
from stack import Stack
from model import Linear_QNet, QTrainer
from helper import plot
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
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        pacman_position = game.pacman.position.asInt()
        pacman_direction = game.pacman.direction.asInt()
        ghost_positions = [ghost.position.asInt() for ghost in game.ghosts]
        ghost_directions = [ghost.direction.asInt() for ghost in game.ghosts]
        state = [pacman_position, pacman_direction, ghost_positions, ghost_directions]
        return state

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # def get_action(self, state):
    #     # random moves: tradeoff exploration / exploitation
    #     self.epsilon = 80 - self.n_games
    #     final_move = [0,0,0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 2)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float)
    #         prediction = self.model(state0)
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1

    #     return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = GameController()
    game.startGame()
    agent = Agent()
    while True:
        game.update()
        state = agent.get_state(game)
        print(game.reward)

        
   

if __name__ == '__main__':       
    train()